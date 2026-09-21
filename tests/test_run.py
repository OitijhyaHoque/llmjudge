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
with open(os.path.join(ROOT, "llmjudge", "prompts", "c2", "user.md"), encoding="utf-8") as _f:
    COLUMNS = re.findall(r"\{([^{}\s]+)\}", _f.read())

# The items file stands in for whatever a caller's rule stage produced. Its group and
# stratum labels are opaque strings here on purpose: this repository never reads them.
#   kept      ctgan_split 0-44    0-39 "accept", 40-44 "pending"
#   positive  ctgan_split 45-57   45-52 sex rejects (gender Male + diag_1 650), 53-57 age
#   test      real_test 0-27
POOL_ROWS = 45 + 13 + 28         # 86
SLOW = {"ctgan_split:3", "ctgan_split:11", "ctgan_split:19", "ctgan_split:27"}
BAD = "ctgan_split:5"            # the item the mock answers with HTTP 400


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


def write_items(root: str) -> str:
    """One JSONL, the only input the judge takes."""
    items = []
    for i in range(58):
        item_id = f"ctgan_split:{i}"
        over = {}
        if item_id in SLOW:
            over["time_in_hospital"] = "14"
        if item_id == BAD:
            over["diag_1"] = "BAD400"
        label = str(i % 2)
        if i < 40:
            group, stratum, weight = "kept", f"accept/label={label}", 1.0
        elif i < 45:
            group, stratum, weight = "kept", f"pending/label={label}", 1.0
        elif i < 53:
            over.update(gender="Male", diag_1="650")
            group, stratum, weight = "positive", "kb.sex_diagnosis", 2.0
        else:
            group, stratum, weight = "positive", "kb.age_diagnosis", 2.0
        items.append({"id": item_id, "group": group, "stratum": stratum, "weight": weight,
                      "fields": base_row(i, **over)})
    for i in range(28):
        items.append({"id": f"real_test:{i}", "group": "test", "stratum": f"label={i % 2}",
                      "fields": base_row(i)})
    path = os.path.join(root, "items.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return path


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
            with open(os.path.join(ROOT, "llmjudge", "prompts", "c2", "system.md")) as f:
                system = f.read().replace('write "verdict" FIRST', 'write "short_reason" FIRST')
            for name, text in (("system.md", system), ("user.md", "{label}")):
                with open(os.path.join(d, name), "w") as f:
                    f.write(text)
            with self.assertRaises(cj.ConfigError):
                cj.load_prompt(d)


class ItemsTests(unittest.TestCase):
    """load_items is the only new parser, and it refuses whole files, never half of one."""

    def _write(self, d: str, *lines: str) -> str:
        path = os.path.join(d, "items.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(l + "\n" for l in lines))
        return path

    def test_order_is_seeded_and_independent_of_file_order(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_items(d)
            a, sha_a, csv_a = cj.load_items(path, 42)
            b, sha_b, csv_b = cj.load_items(path, 42)
            c, _, csv_c = cj.load_items(path, 43)
        self.assertEqual([i.id for i in a], [i.id for i in b])
        self.assertEqual((sha_a, csv_a), (sha_b, csv_b))
        self.assertNotEqual(csv_a, csv_c)                      # a different seed reorders
        self.assertEqual(len(a), POOL_ROWS)
        self.assertEqual({i.id for i in a}, {i.id for i in c})

    def test_optional_labels_default_and_are_never_interpreted(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, json.dumps({"id": "x", "fields": {"a": "1"}}))
            items, _, _ = cj.load_items(path, 42)
        self.assertEqual((items[0].group, items[0].stratum, items[0].weight), ("", "", 1.0))

    def test_a_malformed_file_is_refused_whole(self):
        bad = {
            "duplicate id": ['{"id": "a", "fields": {}}', '{"id": "a", "fields": {}}'],
            "not JSON": ['{"id": "a", "fields": {}}', "{oops"],
            '"id" must be': ['{"fields": {}}'],
            '"fields" must be': ['{"id": "a"}'],
            '"weight" must be': ['{"id": "a", "fields": {}, "weight": "heavy"}'],
            "no items": [],
        }
        for expected, lines in bad.items():
            with tempfile.TemporaryDirectory() as d, self.subTest(expected):
                with self.assertRaises(cj.ConfigError) as cm:
                    cj.load_items(self._write(d, *lines), 42)
                self.assertIn(expected, str(cm.exception))

    def test_every_row_is_checked_against_the_prompt_columns(self):
        """Not only the first one: the row that cannot be rendered used to be found
        mid-run, with the model already warm."""
        with tempfile.TemporaryDirectory() as d:
            path = self._write(
                d,
                json.dumps({"id": "a", "fields": {"age": "1", "gender": "F"}}),
                json.dumps({"id": "b", "fields": {"age": "2"}}),          # no gender
            )
            with self.assertRaises(cj.ConfigError) as cm:
                cj.load_items(path, 42, ("age", "gender"))
            self.assertIn("'b' lacks the fields", str(cm.exception))
            self.assertIn("gender", str(cm.exception))
            self.assertEqual(len(cj.load_items(path, 42, ("age",))[0]), 2)

    def test_a_number_is_a_field_value_like_any_other(self):
        """`{"age": 70}` is valid JSON and a caller will write it sooner or later."""
        self.assertEqual(cj.render_user("age: {age}, n: {n}, x: {x}",
                                        {"age": 70, "n": 5.0, "x": None}),
                         "age: 70, n: 5, x: <missing>")

    def test_missing_items_file(self):
        with self.assertRaises(cj.ConfigError):
            cj.load_items("/nonexistent/items.jsonl", 42)


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
        self.items = write_items(self.root)
        self.out = os.path.join(self.root, "out")
        self.state = MockState()
        self.server = Server(self.state)

    def tearDown(self):
        with contextlib.suppress(Exception):
            self.server.close()
        self.tmp.cleanup()

    def judge(self, *extra: str, endpoints: str | None = None, prompt: str = "c2") -> int:
        argv = ["--prompt", prompt, "--items", self.items,
                "--run-tag", "t", "--out", self.out,
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
        self.assertEqual(len({r["id"] for r in recs}), POOL_ROWS)


class RunTests(RunBase):
    def test_complete_run(self):
        self.assertEqual(self.judge(), cj.EXIT_OK)
        self.assert_one_record_per_row()
        recs = self.results
        bad = [r for r in recs if r["error"]]
        self.assertEqual([(r["id"], r["error"]["class"]) for r in bad],
                         [(BAD, "bad_request")])
        sex = [r["verdict"] for r in recs if r["stratum"] == "kb.sex_diagnosis"]
        self.assertEqual(sex, ["inconsistent"] * 8)
        with open(os.path.join(self.out, "summary.json")) as f:
            summary = json.load(f)
        self.assertEqual((summary["rows"], summary["missing"]), (POOL_ROWS, 0))
        self.assertFalse(os.path.exists(os.path.join(self.out, "judge.pid")))

    def test_an_unrenderable_row_is_recorded_and_does_not_hang(self):
        """Rendering happens inside the request task. When it raised, the in-flight slot
        was never returned, remaining() never reached zero, and the run sat at 85/86
        forever. Any bug in there must cost one row, not the run."""
        doomed = cj.load_items(self.items, 42)[0][-1].id        # last in send order
        with open(self.items, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
        for r in rows:
            if r["id"] == doomed:
                r["fields"]["diag_2"] = "EXPLODE"
        with open(self.items, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

        real = cj.render_user

        def boom(template, row):
            if row.get("diag_2") == "EXPLODE":
                raise ValueError("deliberate")
            return real(template, row)

        cj.render_user = boom
        rc: list[int] = []
        th = threading.Thread(target=lambda: rc.append(self.judge()), daemon=True)
        try:
            th.start()
            th.join(60)
        finally:
            cj.render_user = real
        self.assertFalse(th.is_alive(), "the run never finished")
        self.assertEqual(rc, [cj.EXIT_OK])
        self.assert_one_record_per_row()
        internal = [r for r in self.results if (r["error"] or {}).get("class") == "internal"]
        self.assertEqual([r["id"] for r in internal], [doomed])
        self.assertIn("ValueError", internal[0]["error"]["detail"])

    def test_slow_rows_do_not_hold_the_others(self):
        self.state.delay, self.state.slow_seconds = 0.02, 1.5
        endpoints = write_endpoints(self.root, self.server.port, concurrency=8, max_concurrency=8)
        t0 = time.monotonic()
        self.assertEqual(self.judge(endpoints=endpoints), cj.EXIT_OK)
        wall = time.monotonic() - t0
        # 4 slow rows in batches of 8 would cost >= 4 x 1.5 s; a continuous window ~1.5 s.
        self.assertLess(wall, 4.0)
        self.assert_one_record_per_row()
        order = [r["id"] for r in self.results]
        self.assertEqual(set(order[-len(SLOW):]), SLOW)

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
