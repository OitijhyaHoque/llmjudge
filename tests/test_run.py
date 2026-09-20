"""llmjudge/run.py against a threaded mock vLLM server.

Covers what the stream runs got wrong: a slow row must not hold the others, an endpoint
that dies mid-run must lose and duplicate nothing, a restart must resume, the request
quota must stop an account, and a server that ignores the schema must be refused. The
mock's verdicts are a fixed rule; nothing here says anything about a real model.

    python3 -m unittest tests.test_run             (from the repository root)
"""

from __future__ import annotations

import collections
import contextlib
import csv
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from llmjudge import run as cj  # noqa: E402

MODEL = "mock/medgemma"
with open(os.path.join(ROOT, "prompts", "c2", "user.md"), encoding="utf-8") as _f:
    COLUMNS = re.findall(r"\{([^{}\s]+)\}", _f.read())

# ctgan_split: 0-39 accept, 40-44 pending, 45-52 sex rejects, 53-57 age rejects, 58-59 other.
# real_test: 0-27 accept, 28-29 reject. Full custom pool = 45 + 13 + 28 = 86 rows.
POOL_ROWS = 86
SLOW = {3, 11, 19, 27}           # ctgan rows with time_in_hospital 14
BAD = 5                          # ctgan row the mock answers with HTTP 400


# --------------------------------------------------------------------------------------
# mock server


class MockState:
    def __init__(self):
        self.lock = threading.Lock()
        self.dead = False            # read the request, answer nothing, drop the socket
        self.delay = 0.0
        self.slow_seconds = 0.0
        self.fail_503 = 0
        self.ignore_schema = False
        self.chat_calls = 0
        self.concurrent = 0
        self.max_concurrent = 0


def make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _json(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if state.dead:
                self.close_connection = True
                return
            if self.path.endswith("/models"):
                return self._json(200, {"object": "list", "data": [{"id": MODEL}]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if state.dead:
                self.close_connection = True
                return
            with state.lock:
                state.chat_calls += 1
                state.concurrent += 1
                state.max_concurrent = max(state.max_concurrent, state.concurrent)
                fail = state.fail_503 > 0
                state.fail_503 -= fail
            try:
                if fail:
                    return self._json(503, {"error": "busy"})
                user = req["messages"][-1]["content"]
                if "diag_1: BAD400" in user:
                    return self._json(400, {"error": "bad request"})
                time.sleep(state.delay
                           + (state.slow_seconds if "time_in_hospital: 14" in user else 0))
                if state.dead:
                    self.close_connection = True
                    return
                schema = req["response_format"]["json_schema"]["schema"]
                enum = schema["properties"]["verdict"]["enum"]
                if state.ignore_schema:
                    obj = {"verdict": "consistent", "short_reason": "no conflict"}
                else:
                    verdict = enum[0] if len(enum) == 1 else (
                        "inconsistent" if "gender: Male" in user and "diag_1: 650" in user
                        else "consistent")
                    values = {"verdict": verdict, "short_reason": "no conflict"}
                    obj = {k: values[k] for k in schema["properties"]}
                self._json(200, {
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": json.dumps(obj)}}],
                    "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": 12}})
            finally:
                with state.lock:
                    state.concurrent -= 1

    return Handler


class Server:
    def __init__(self, state: MockState, port: int = 0):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


# --------------------------------------------------------------------------------------
# fixture


def base_row(i: int, **over) -> dict:
    r = {c: "No" for c in COLUMNS}
    r.update(label=str(i % 2), race="Caucasian", gender="Female", age="[60-70)",
             admission_type_id="1", admission_source_id="7", discharge_disposition_id="1",
             time_in_hospital="3", payer_code="", medical_specialty="",
             num_lab_procedures="40", num_procedures="1", num_medications="10",
             number_diagnoses="7", number_outpatient="0", number_emergency="0",
             number_inpatient="0", diag_1="428", diag_2="250.02", diag_3="401",
             max_glu_serum="", A1Cresult="", insulin="Steady", change="No",
             diabetesMed="Yes")
    r.update(over)
    return r


def write_arm(run: str, arm: str, rows: list[dict], decisions: list[tuple[str, str]]):
    os.makedirs(os.path.join(run, "normalize", arm))
    os.makedirs(os.path.join(run, "rules", arm))
    with open(os.path.join(run, "normalize", arm, "table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(run, "rules", arm, "decisions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "decision", "action", "stage", "kept", "checks",
                    "defect_classes", "columns"])
        for i, (decision, checks) in enumerate(decisions):
            w.writerow([i, decision, "use", "clean", "0" if decision == "reject" else "1",
                        checks, "", ""])


def write_fixture(root: str) -> str:
    run = os.path.join(root, "run")
    rows, decs = [], []
    for i in range(60):
        over = {}
        if i in SLOW:
            over["time_in_hospital"] = "14"
        if i == BAD:
            over["diag_1"] = "BAD400"
        if i < 40:
            decs.append(("accept", ""))
        elif i < 45:
            decs.append(("pending", "kb.duplicate_class"))
        elif i < 53:
            decs.append(("reject", "kb.sex_diagnosis"))
            over.update(gender="Male", diag_1="650")
        elif i < 58:
            decs.append(("reject", "kb.age_diagnosis"))
        else:
            decs.append(("reject", "book.change_no_but_titrated"))
        rows.append(base_row(i, **over))
    write_arm(run, "ctgan_split", rows, decs)
    write_arm(run, "real_test", [base_row(i) for i in range(30)],
              [("accept", "")] * 28 + [("reject", "range.number_outpatient")] * 2)
    return run


def write_endpoints(root: str, port: int, quota: int = 0, concurrency: int = 4,
                    max_concurrency: int = 16) -> str:
    env = os.path.join(root, ".env")
    with open(env, "w") as f:
        f.write(f"MOCK_URL=http://127.0.0.1:{port}/v1\n")
    path = os.path.join(root, "endpoints.toml")
    with open(path, "w") as f:
        f.write(f'quota_file = "{root}/quota.json"\n\n[[endpoint]]\nname = "mock"\n'
                f'env_file = "{env}"\nurl_key = "MOCK_URL"\nmodel = "{MODEL}"\n'
                f"concurrency = {concurrency}\nmin_concurrency = 1\n"
                f'max_concurrency = {max_concurrency}\naccount = "acct"\nquota = {quota}\n')
    return path


def read_lines(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------------------
# pure functions


class ParserTests(unittest.TestCase):
    ORDER = ("verdict", "short_reason")

    def test_valid_reply_and_key_order(self):
        ok = cj.parse_strict('{"verdict": "unsure", "short_reason": "x"}', self.ORDER)
        self.assertEqual((ok["verdict"], ok["key_order_ok"]), ("unsure", True))
        swapped = cj.parse_strict('{"short_reason": "x", "verdict": "unsure"}', self.ORDER)
        self.assertFalse(swapped["key_order_ok"])

    def test_everything_else_is_a_parse_error(self):
        for bad in ['{"verdict": "maybe", "short_reason": "x"}',
                    '{"verdict": "consistent"}',
                    '{"verdict": "consistent", "short_reason": "x", "reasoning": "y"}',
                    '["consistent"]', 'verdict: consistent', '', None]:
            self.assertIn("parse_error", cj.parse_strict(bad, self.ORDER), bad)


class PromptTests(unittest.TestCase):
    def test_schema_order_follows_each_prompt(self):
        self.assertEqual(cj.load_prompt("c2")["order"], ("verdict", "short_reason"))
        rf = cj.load_prompt("c2-reason-first")
        self.assertEqual(rf["order"], ("short_reason", "verdict"))
        self.assertEqual(list(rf["schema"]["properties"]), ["short_reason", "verdict"])

    def test_first_instruction_contradicting_the_example_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(ROOT, "prompts", "c2", "system.md")) as f:
                system = f.read().replace('write "verdict" FIRST', 'write "short_reason" FIRST')
            for name, text in (("system.md", system), ("user.md", "{label}")):
                with open(os.path.join(d, name), "w") as f:
                    f.write(text)
            with self.assertRaises(cj.ConfigError):
                cj.load_prompt(d)


class PoolTests(unittest.TestCase):
    def test_allocation(self):
        self.assertEqual(cj.allocate({"a": 90, "b": 10}, 10, "proportional"), {"a": 9, "b": 1})
        self.assertEqual(cj.allocate({"a": 90, "b": 3}, 10, "equal"), {"a": 7, "b": 3})
        self.assertEqual(cj.allocate({"a": 5, "b": 3}, 0, "equal"), {"a": 5, "b": 3})

    def test_pool_is_deterministic_and_small_pools_nest_in_large_ones(self):
        with tempfile.TemporaryDirectory() as d:
            run = write_fixture(d)
            small = cj.build_pool(run, {"kept": 5, "positive": 4, "test": 4}, 42)
            again = cj.build_pool(run, {"kept": 5, "positive": 4, "test": 4}, 42)
            large = cj.build_pool(run, {"kept": 10, "positive": 8, "test": 8}, 42)
        self.assertEqual(small[3], again[3])
        ids = lambda pool: {(it.group, it.arm, it.index) for it in pool[0]}  # noqa: E731
        self.assertLessEqual(ids(small), ids(large))
        strata = [it.stratum for it in small[0] if it.group == "positive"]
        self.assertEqual(sorted(strata), ["kb.age_diagnosis"] * 2 + ["kb.sex_diagnosis"] * 2)


class TunerTests(unittest.TestCase):
    def _round(self, t: cj.Tuner, seconds: float):
        t.t0 = time.monotonic() - seconds
        for _ in range(max(8, t.limit)):
            t.on_success(0.1, binding=True, timeout=120)

    def test_grows_while_throughput_rises_then_steps_back(self):
        t = cj.Tuner(8, 1, 64)
        self._round(t, 1.0)                        # 8/s, first measurement
        self.assertEqual(t.limit, 10)
        self._round(t, 1.0)                        # 10/s, better
        self.assertEqual(t.limit, 12)
        self._round(t, 2.0)                        # 6/s, worse: back to 10 and hold
        self.assertEqual((t.limit, t.mode), (10, "hold"))

    def test_overload_halves_at_most_once_per_window(self):
        t = cj.Tuner(32, 2, 64)
        t.on_overload()
        t.on_overload()
        self.assertEqual(t.limit, 16)


# --------------------------------------------------------------------------------------
# end to end


class RunBase(unittest.TestCase):
    """A fixture run directory, a mock server, and a way to call the judge."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.run_dir = write_fixture(self.root)
        self.out = os.path.join(self.root, "out")
        self.state = MockState()
        self.server = Server(self.state)

    def tearDown(self):
        with contextlib.suppress(Exception):
            self.server.close()
        self.tmp.cleanup()

    def judge(self, *extra: str, endpoints: str | None = None, prompt: str = "c2") -> int:
        argv = ["--prompt", prompt, "--pool", "full", "--sizes", "kept=0,positive=0,test=0",
                "--run-dir", self.run_dir, "--run-tag", "t", "--out", self.out,
                "--endpoints", endpoints or write_endpoints(self.root, self.server.port),
                "--timeout", "10", "--probe-min", "0.1", "--probe-max", "0.4",
                "--give-up-after", "20", "--progress-every", "0.5", "--backoff-max", "0.2",
                *extra]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cj.main(argv)

    @property
    def results(self) -> list[dict]:
        return read_lines(os.path.join(self.out, "results.jsonl"))

    def assert_one_record_per_row(self):
        recs = self.results
        self.assertEqual(len(recs), POOL_ROWS)
        self.assertEqual(len({r["key"] for r in recs}), POOL_ROWS)
        self.assertEqual(len({(r["arm"], r["index"]) for r in recs}), POOL_ROWS)


class RunTests(RunBase):
    def test_complete_run(self):
        self.assertEqual(self.judge(), cj.EXIT_OK)
        self.assert_one_record_per_row()
        recs = self.results
        bad = [r for r in recs if r["error"]]
        self.assertEqual([(r["arm"], r["index"], r["error"]["class"]) for r in bad],
                         [("ctgan_split", BAD, "bad_request")])
        sex = [r["verdict"] for r in recs if r["rules_checks"] == "kb.sex_diagnosis"]
        self.assertEqual(sex, ["inconsistent"] * 8)
        with open(os.path.join(self.out, "summary.json")) as f:
            summary = json.load(f)
        self.assertEqual((summary["rows"], summary["missing"]), (POOL_ROWS, 0))
        self.assertFalse(os.path.exists(os.path.join(self.out, "judge.pid")))

    def test_slow_rows_do_not_hold_the_others(self):
        self.state.delay, self.state.slow_seconds = 0.02, 1.5
        endpoints = write_endpoints(self.root, self.server.port, concurrency=8, max_concurrency=8)
        t0 = time.monotonic()
        self.assertEqual(self.judge(endpoints=endpoints), cj.EXIT_OK)
        wall = time.monotonic() - t0
        # 4 slow rows in batches of 8 would cost >= 4 x 1.5 s; a continuous window ~1.5 s.
        self.assertLess(wall, 4.0)
        self.assert_one_record_per_row()
        order = [r["index"] for r in self.results if r["arm"] == "ctgan_split"]
        self.assertTrue(all(i in SLOW for i in order[-len(SLOW):]))

    def test_endpoint_dies_mid_run_and_nothing_is_lost_or_duplicated(self):
        self.state.delay = 0.05
        endpoints = write_endpoints(self.root, self.server.port)
        port = self.server.port
        rc: list[int] = []
        th = threading.Thread(target=lambda: rc.append(self.judge(endpoints=endpoints)))
        th.start()
        deadline = time.monotonic() + 20
        while len(self.results) < 15 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.state.dead = True                      # in-flight requests get no answer
        self.server.close()                         # and new connections are refused
        time.sleep(1.0)
        self.server = Server(self.state, port)
        self.state.dead = False
        th.join(60)
        self.assertFalse(th.is_alive())
        self.assertEqual(rc, [cj.EXIT_OK])
        self.assert_one_record_per_row()
        with open(os.path.join(self.out, "endpoints.json")) as f:
            events = " ".join(json.load(f)["endpoints"][0]["events"])
        self.assertIn("DOWN", events)
        self.assertIn("down)", events)              # "up ... (after Xs down)"

    def test_resume_continues_and_refuses_a_changed_prompt(self):
        self.assertEqual(self.judge("--limit", "10"), cj.EXIT_OK)
        self.assertEqual(len(self.results), 10)
        calls = self.state.chat_calls
        self.assertEqual(self.judge(), cj.EXIT_OK)
        self.assert_one_record_per_row()
        self.assertEqual(self.state.chat_calls - calls, 1 + POOL_ROWS - 10)   # canary + rest
        self.assertEqual(self.judge(prompt="c2-reason-first"), cj.EXIT_CONFIG)

    def test_retry_errors_resends_only_error_rows(self):
        self.assertEqual(self.judge(), cj.EXIT_OK)
        calls = self.state.chat_calls
        self.assertEqual(self.judge("--retry-errors"), cj.EXIT_OK)
        self.assertEqual(self.state.chat_calls - calls, 2)                    # canary + row 5
        self.assertEqual(len(self.results), POOL_ROWS + 1)

    def test_transient_503s_are_retried(self):
        self.state.fail_503 = 3
        self.assertEqual(self.judge(), cj.EXIT_OK)
        self.assert_one_record_per_row()
        self.assertEqual(sum(bool(r["error"]) for r in self.results), 1)      # only row 5

    def test_quota_stops_the_account(self):
        endpoints = write_endpoints(self.root, self.server.port, quota=10, concurrency=1,
                                    max_concurrency=1)
        self.assertEqual(self.judge(endpoints=endpoints), cj.EXIT_INCOMPLETE)
        with open(os.path.join(self.root, "quota.json")) as f:
            used = sum(json.load(f).values())
        self.assertEqual(used, 10)
        self.assertEqual(len(self.results), 8)      # 10 minus /v1/models and the canary

    def test_server_ignoring_the_schema_is_refused(self):
        self.state.ignore_schema = True
        self.assertEqual(self.judge(), cj.EXIT_INCOMPLETE)
        self.assertEqual(self.results, [])
        with open(os.path.join(self.out, "endpoints.json")) as f:
            self.assertEqual(json.load(f)["endpoints"][0]["state"], "refused")


class ClassifyTests(unittest.TestCase):
    def test_cloudflare_and_ngrok_statuses(self):
        self.assertEqual(cj.classify(524, None), "timeout")
        for status in (502, 520, 522, 530):
            self.assertEqual(cj.classify(status, None), "down", status)
        self.assertEqual(cj.classify(404, "ERR_NGROK_3200"), "down")
        self.assertEqual(cj.classify(404, None), "bad_request")
        self.assertEqual(cj.classify(401, None), "auth")


class HotSwapTests(RunBase):
    """Endpoints that appear, or move, while a run is going."""

    def write_two(self, port_a: int) -> tuple[str, str]:
        env = os.path.join(self.root, ".env")
        with open(env, "w") as f:
            f.write(f"MOCK_URL=http://127.0.0.1:{port_a}/v1\n")
        path = os.path.join(self.root, "endpoints.toml")
        with open(path, "w") as f:
            f.write(f'quota_file = "{self.root}/quota.json"\n')
            for name, key in (("a", "MOCK_URL"), ("b", "MOCK_URL_2")):
                f.write(f'\n[[endpoint]]\nname = "{name}"\nenv_file = "{env}"\n'
                        f'url_key = "{key}"\nmodel = "{MODEL}"\nconcurrency = 2\n'
                        f"min_concurrency = 1\nmax_concurrency = 2\nquota = 0\n")
        return env, path

    def wait_for_rows(self, n: int) -> None:
        deadline = time.monotonic() + 20
        while len(self.results) < n and time.monotonic() < deadline:
            time.sleep(0.02)

    def test_second_endpoint_joins_when_its_url_appears(self):
        self.state.delay = 0.05
        state_b = MockState()
        state_b.delay = 0.05
        server_b = Server(state_b)
        try:
            env, endpoints = self.write_two(self.server.port)
            rc: list[int] = []
            th = threading.Thread(target=lambda: rc.append(self.judge(endpoints=endpoints)))
            th.start()
            self.wait_for_rows(10)
            with open(env, "a") as f:
                f.write(f"MOCK_URL_2=http://127.0.0.1:{server_b.port}/v1\n")
            th.join(60)
        finally:
            server_b.close()
        self.assertEqual(rc, [cj.EXIT_OK])
        self.assert_one_record_per_row()
        by_ep = collections.Counter(r["endpoint"] for r in self.results)
        self.assertGreater(by_ep["b"], 5, by_ep)
        self.assertGreater(by_ep["a"], 10, by_ep)

    def test_running_judge_follows_a_changed_url(self):
        self.state.delay = 0.05
        env = os.path.join(self.root, ".env")
        endpoints = write_endpoints(self.root, self.server.port)
        rc: list[int] = []
        th = threading.Thread(target=lambda: rc.append(self.judge(endpoints=endpoints)))
        th.start()
        self.wait_for_rows(10)
        self.state.dead = True                 # the old notebook goes away
        self.server.close()
        self.server = Server(self.state)       # the restarted one, on a new address
        self.state.dead = False
        with open(env, "w") as f:
            f.write(f"MOCK_URL=http://127.0.0.1:{self.server.port}/v1\n")
        th.join(60)
        self.assertEqual(rc, [cj.EXIT_OK])
        self.assert_one_record_per_row()
        with open(os.path.join(self.out, "endpoints.json")) as f:
            events = " ".join(json.load(f)["endpoints"][0]["events"])
        self.assertIn("changed in the env file", events)


if __name__ == "__main__":
    unittest.main()
