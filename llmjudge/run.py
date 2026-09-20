#!/usr/bin/env python3
"""Whole-row judge over vLLM's OpenAI-compatible API, with or without guided decoding.

Built from `scripts/c2_judge.py`; the pool, the endpoint pool, resume and the failure
handling are unchanged. What is new is `--guided off`, for prompts that reason in plain text
before they answer (`prompts/c3-reasoning`), which no JSON schema can hold.

    --guided on    the default. `response_format` carries the two-key schema and the reply
                   must be exactly that object, in the schema's key order. This is c2_judge's
                   behaviour, down to the resume keys, so a c2 run resumes under this script.
    --guided off   no `response_format` at all. The reply may carry reasoning in plain text;
                   the verdict is the LAST JSON object with a `verdict` key, so prose and code
                   fences around it are tolerated, and the text before it is kept as
                   `reasoning`. --max-tokens then defaults to 1024, not 96, because the
                   reasoning has to fit: a truncated reply is recorded as an error, never a
                   verdict. 1024 is measured, not guessed -- see below.

Sizing --max-tokens without a grammar. On MedGemma 27B IT the only thing that held the reply
short in earlier runs was the grammar, so the budget has to come from the unguided runs:

    run                    guided  completion_tokens p50 / p90 / max
    runs/c1/ngrok-27b-c1r     yes             537 /  780 /  931
    runs/c1/ngrok-27b-c1r-stream  no         1824 / 2232 / 2476   (thinking trigger on)

The unguided run is an upper bound in two ways: it ran c1r, whose reasoning field walks 45
columns and emits a findings array, and it appended the thinking trigger, so most of those
tokens are a `<unused94>thought` span. c3-reasoning asks for four lines and at most 120 words
(~250 tokens) and does not trigger thinking. 1024 is ~4x that, and still well inside the
--timeout 95 s budget; 2048+ would be caught by the timeout, not by max_tokens, and cost a
retry. Check `truncated` in summary.json after a pilot before raising it.

Plan: `notes/i-0093-02/newplan.md` §3-§6. One row per request: the whole row rendered into
`prompts/<prompt>/user.md`, temperature 0. The model never sees the arm or the row index.

Throughput
    Nothing is sent in batches. Each endpoint keeps a window of in-flight requests and
    starts the next one the moment any finishes, so a 100 s row holds one slot while the
    rest keep flowing; vLLM batches whatever is in flight on its side. The window adapts:
    it grows (+25%) while measured throughput keeps rising, steps back when a step bought
    nothing (the server is saturated), and halves on 429s and timeouts. Bounds come from
    `configs/c2_endpoints.toml`. With `metrics = true` it also stops growing while vLLM
    reports queued requests; that is off by default, because every poll through ngrok
    counts against the monthly request quota.

Failures
    endpoint down   connection errors, ngrok errors (ERR_NGROK_*), 502/503/504, Cloudflare
                    520-523/525-527/530: the row goes back on the queue, the endpoint
                    pauses, `/v1/models` + a canary are probed with backoff (--probe-min ..
                    --probe-max), and it resumes on its own. In shard mode the other
                    endpoint drains the queue meanwhile. Five timeouts or 5xx in a row also
                    pause it, since a dead engine can still answer `/v1/models`.
    new URL         while an endpoint is down its env file is re-read every few seconds,
                    and a changed URL is probed at once. A Colab restart gets a new
                    trycloudflare hostname: update .env and the running judge follows. An
                    endpoint whose URL key is not in .env yet waits the same way, so a
                    second notebook can join mid-run (its entry must be in the endpoints
                    file from the start).
    401 / 403       the env file is re-read (keys rotate) and the row retried once.
    429, timeout    also Cloudflare 524 (origin silent ~100 s): window halved, row retried
                    with jittered backoff, at most --max-attempts, then recorded as an
                    error.
    400 / 422       recorded as an error for that row; 20 in a row refuses the endpoint.
    bad reply       non-JSON or wrong keys: retried once, then recorded with parse_error.
                    finish_reason=length is recorded as an error, never as a verdict.
    quota           requests are counted per ngrok account per calendar month (UTC) in
                    `quota_file`; an account stops at its `quota`.
    gives up        only when every endpoint with work left has been down for
                    --give-up-after seconds, or none is usable.

Preflight, per endpoint, before any row: `/v1/models` must list the configured model, and a
canary request must come back sound. Guided, the canary's schema only allows verdict
"canary-ok" and the reply must be exactly that in the schema's key order, so a server that
silently ignores `response_format` is refused. Unguided there is nothing to enforce, so the
canary only has to parse; a canary that does not parse is a warning, and the breaker that
refuses an endpoint whose first 20 replies all fail parsing does the rest.

Resume
    `<out>/results.jsonl` is append-only, one line per final answer, flushed per line and
    fsync'd every 50. A restart skips every row already recorded; `--retry-errors` re-sends
    rows whose latest record is an error or a parse error. The resume key is (input table
    sha, arm, row index, model, prompt sha, schema sha or "guided=off"), plus the endpoint in
    compare mode.
    `<out>/run.json` pins prompt, schema, decoding, pool and inputs; a restart with any of
    them changed is refused. Ctrl-C once: stop sending, let in-flight requests finish,
    write the summary. Twice: exit now (every recorded line is already on disk).

Pools (row sources: `<run-dir>/normalize/<arm>/table.csv` + `rules/<arm>/decisions.csv`)
    kept       ctgan_split rows the hard rules kept      pilot 50   full all (52,539)
    positive   ctgan_split rows rejected by kb.sex_diagnosis / kb.age_diagnosis
                                                         pilot 25   full 500
    test       real_test rows the hard rules kept        pilot 25   full 2,000
    `kept` and `test` are stratified proportionally ((decision, label) and label), and
    `positive` equally over the two checks, each stratum shuffled by --seed. The pilot is
    a subset of the full pool. Rows are sent in one seeded shuffled order, so any prefix
    of a run is a random sample of the pool. Every row carries weight = stratum size /
    rows taken from it.

Modes
    shard      one shared queue, every endpoint pulls from it; all must serve one model.
    compare    every row to every endpoint (e.g. MedGemma vs a general model). With two
               replicas of one model, `--mode compare --limit 200` is the agreement check.

    python3 -m llmjudge --prompt c3 --pool pilot --run-tag c3-pilot-01 --dry-run
    python3 -m llmjudge --prompt c3 --pool pilot --run-tag c3-pilot-01
    python3 -m llmjudge --prompt c3-reasoning --guided off --pool pilot \\
        --run-tag c3-reasoning-pilot-01
    nohup python3 -m llmjudge --prompt c3-reasoning --guided off --pool full \\
        --run-tag c3r-full-01 \\
        >> runs/split-ctgan-e300-gpu/c2/c3r-full-01.log 2>&1 &

Outputs in `<out>` (default `<run-dir>/c2/<run-tag>`): pool.csv, run.json, results.jsonl,
summary.json, endpoints.json, prompt_example.txt, request_example.json, judge.pid.
Exit codes: 0 every planned row recorded, 2 configuration refused, 3 stopped incomplete.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
import random
import re
import signal
import sys
import time
import tomllib

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

from .prompts import render_user
from .tables import iter_rows

VERDICTS = ("consistent", "inconsistent", "unsure")
KEYS = ("verdict", "short_reason")
CANARY = "canary-ok"
POSITIVE_CHECKS = ("kb.sex_diagnosis", "kb.age_diagnosis")
HEADERS = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "1"}
NGROK_CODE = re.compile(r"ERR_NGROK_\d+")
FIRST = re.compile(r'write "(\w+)" FIRST')
WAITING = re.compile(r"^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.M)
THOUGHT = re.compile(r"<unused94>.*?(?:<unused95>|$)", re.S)   # MedGemma's thinking span

GROUPS = {
    "kept": {"arm": "ctgan_split", "alloc": "proportional",
             "select": lambda d: d["kept"] == "1",
             "stratum": lambda d, label: f"{d['decision']}/label={label}"},
    "positive": {"arm": "ctgan_split", "alloc": "equal",
                 "select": lambda d: d["decision"] == "reject" and d["checks"] in POSITIVE_CHECKS,
                 "stratum": lambda d, label: d["checks"]},
    "test": {"arm": "real_test", "alloc": "proportional",
             "select": lambda d: d["kept"] == "1",
             "stratum": lambda d, label: f"label={label}"},
}
SIZES = {"pilot": {"kept": 50, "positive": 25, "test": 25},
         "full": {"kept": 0, "positive": 500, "test": 2000}}      # 0 = every eligible row

EXIT_OK, EXIT_CONFIG, EXIT_INCOMPLETE = 0, 2, 3
POOL_FIELDS = ["order", "group", "stratum", "arm", "index", "label", "rules_decision",
               "rules_checks", "weight"]


class ConfigError(Exception):
    """A refusal: the run cannot start as configured."""


# --------------------------------------------------------------------------------------
# small helpers


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    """Relative paths are relative to the repository root, wherever this is started from."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(ROOT, path))


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT)


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def fmt_dur(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def read_env(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.removeprefix("export ")
            k, sep, v = line.partition("=")
            if sep:
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


# --------------------------------------------------------------------------------------
# prompt, schema, parsing


def load_prompt(arg: str) -> dict:
    d = resolve(arg if os.sep in arg else os.path.join("prompts", arg))
    try:
        with open(os.path.join(d, "system.md"), encoding="utf-8") as f:
            system = f.read().strip()
        with open(os.path.join(d, "user.md"), encoding="utf-8") as f:
            user = f.read().strip()
    except FileNotFoundError as e:
        raise ConfigError(f"prompt {arg!r}: {e}") from e
    order = key_order(system)
    schema = build_schema(order)
    return {"name": os.path.basename(d.rstrip(os.sep)), "dir": d, "system": system,
            "user": user, "order": order, "schema": schema,
            "prompt_sha": sha(system + "\0" + user),
            "schema_sha": sha(json.dumps(schema, separators=(",", ":")))}


def key_order(system: str) -> tuple[str, ...]:
    """The output key order the prompt asks for, which becomes the schema's order.

    Under guided decoding keys are generated in schema order, so a reason-first prompt
    with a verdict-first schema would silently be the no-reasoning arm. The order is read
    from the prompt's one JSON example line, and a `write "X" FIRST` instruction that
    disagrees with it is refused.
    """
    examples = [ln for ln in system.splitlines()
                if ln.lstrip().startswith("{") and all(f'"{k}"' in ln for k in KEYS)]
    if len(examples) != 1:
        raise ConfigError(f"expected one JSON example line with keys {KEYS} in the system "
                          f"prompt, found {len(examples)}")
    order = tuple(sorted(KEYS, key=lambda k: examples[0].index(f'"{k}"')))
    first = FIRST.findall(system)
    if first and first[-1] != order[0]:
        raise ConfigError(f'the prompt says write "{first[-1]}" FIRST but its JSON example '
                          f'starts with "{order[0]}"')
    return order


def build_schema(order: tuple[str, ...], verdicts: tuple[str, ...] = VERDICTS) -> dict:
    props = {"verdict": {"type": "string", "enum": list(verdicts)},
             "short_reason": {"type": "string"}}
    return {"type": "object", "properties": {k: props[k] for k in order},
            "required": list(order), "additionalProperties": False}


def parse_strict(content: str | None, order: tuple[str, ...]) -> dict:
    """json.loads, exactly the two keys, a valid verdict. No fallbacks: with a schema, a
    reply that needs one means the server is not enforcing it."""
    try:
        obj = json.loads(content or "")
    except ValueError as e:
        return {"parse_error": f"not JSON: {str(e)[:80]}"}
    if not isinstance(obj, dict):
        return {"parse_error": f"JSON {type(obj).__name__}, not an object"}
    if set(obj) != set(KEYS):
        return {"parse_error": f"keys {sorted(obj)}"}
    if obj["verdict"] not in VERDICTS:
        return {"parse_error": f"verdict {obj['verdict']!r}"}
    if not isinstance(obj["short_reason"], str):
        return {"parse_error": "short_reason is not a string"}
    return {"verdict": obj["verdict"], "short_reason": obj["short_reason"],
            "key_order_ok": tuple(obj) == tuple(order)}


def parse_tolerant(content: str | None, order: tuple[str, ...]) -> dict:
    """Without a grammar: the LAST JSON object carrying a `verdict`, and the text before it.

    Scanning for the last such object, rather than the span between the first `{` and the
    last `}`, tolerates plain-text reasoning and code fences around the answer, the way
    c1_judge.parse_verdict does. Whatever the model wrote before that object is kept as
    `reasoning`. MedGemma's `<unused94>thought` span is dropped, so draft JSON inside it
    cannot win over the answer after it -- unless the span never closes, which is the one
    case `parse_note` records. Still exactly the two keys and a valid verdict: this is the reply the prompt
    asks for, only without a grammar to guarantee it, so a reply that misses is a parse error
    and never a guessed verdict.
    """
    raw = content or ""
    body, note = THOUGHT.sub("", raw), None
    if raw.rfind("<unused94>") > raw.rfind("<unused95>"):
        # A thought span that never closes runs to the end of the text, so THOUGHT strips
        # the answer with it. Scan the whole reply instead, markers removed. This happened
        # on 58 of 121 replies in runs/c1/ngrok-27b-c1r-stream.
        body = raw.replace("<unused94>", "").replace("<unused95>", "")
        note = "json inside unclosed thought"
    decoder, found, at = json.JSONDecoder(), None, 0
    for m in re.finditer(r"\{", body):
        try:
            obj, _ = decoder.raw_decode(body, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            found, at = obj, m.start()
    if found is None:
        return {"parse_error": f"no JSON object with a verdict: {body.strip()[:80]!r}"}
    if set(found) != set(KEYS):
        return {"parse_error": f"keys {sorted(found)}"}
    verdict = found["verdict"]
    if isinstance(verdict, str):
        verdict = verdict.strip().lower()
    if verdict not in VERDICTS:
        return {"parse_error": f"verdict {found['verdict']!r}"}
    if not isinstance(found["short_reason"], str):
        return {"parse_error": "short_reason is not a string"}
    return {"verdict": verdict, "short_reason": found["short_reason"],
            "reasoning": body[:at].strip() or None, "parse_note": note,
            "key_order_ok": tuple(found) == tuple(order)}


def build_payload(model: str, prompt: dict, user_text: str, decoding: dict,
                  extra: dict | None = None, schema: dict | None = None,
                  name: str = "c2_verdict", guided: bool = True) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": prompt["system"]},
                     {"role": "user", "content": user_text}],
        "temperature": decoding["temperature"], "seed": decoding["seed"],
        "max_tokens": decoding["max_tokens"],
    }
    if guided:
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": name, "schema": schema or prompt["schema"], "strict": True}}
    payload.update(extra or {})
    return payload


# --------------------------------------------------------------------------------------
# pool


class Item:
    __slots__ = ("group", "stratum", "arm", "index", "label", "decision", "checks", "weight",
                 "row", "key", "attempts", "parse_retries", "auth_retried")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        self.attempts, self.parse_retries, self.auth_retried = 0, 0, False

    def clone(self, key: str) -> Item:
        return Item(**{k: getattr(self, k) for k in self.__slots__ if k != "key"}, key=key)


def allocate(sizes: dict[str, int], n: int, alloc: str) -> dict[str, int]:
    """Rows to take per stratum. n <= 0 or n >= total takes everything."""
    total = sum(sizes.values())
    keys = sorted(sizes)
    if n <= 0 or n >= total:
        return dict(sizes)
    if alloc == "equal":
        take, left = dict.fromkeys(keys, 0), n
        while left:
            moved = False
            for k in keys:
                if left and take[k] < sizes[k]:
                    take[k] += 1
                    left -= 1
                    moved = True
            if not moved:
                break
        return take
    exact = {k: n * sizes[k] / total for k in keys}
    take = {k: min(sizes[k], math.floor(exact[k])) for k in keys}
    left = n - sum(take.values())
    for k in sorted(keys, key=lambda k: (-(exact[k] - math.floor(exact[k])), k)):
        if left <= 0:
            break
        if take[k] < sizes[k]:
            take[k] += 1
            left -= 1
    return take


def build_pool(run_dir: str, sizes: dict[str, int], seed: int
               ) -> tuple[list[Item], dict[str, list[str]], dict, str]:
    """-> (items in send order, header per arm, input provenance, pool.csv text)."""
    by_arm: dict[str, list[str]] = collections.defaultdict(list)
    for g, spec in GROUPS.items():
        by_arm[spec["arm"]].append(g)

    items: list[Item] = []
    headers: dict[str, list[str]] = {}
    inputs: dict[str, dict] = {}
    for arm, groups in sorted(by_arm.items()):
        table = os.path.join(run_dir, "normalize", arm, "table.csv")
        decisions = os.path.join(run_dir, "rules", arm, "decisions.csv")
        for p in (table, decisions):
            if not os.path.exists(p):
                raise ConfigError(f"missing input {p}")
        inputs[arm] = {"table": rel(table), "table_sha": file_sha(table),
                       "decisions": rel(decisions), "decisions_sha": file_sha(decisions)}

        strata: dict[str, dict[str, list[tuple]]] = {
            g: collections.defaultdict(list) for g in groups}
        try:
            for i, (d, row) in enumerate(zip(iter_rows(decisions), iter_rows(table),
                                             strict=True)):
                if int(d["index"]) != i:
                    raise ConfigError(f"{decisions}: line {i} has index {d['index']}")
                label = row.get("label", "")
                for g in groups:
                    if GROUPS[g]["select"](d):
                        strata[g][GROUPS[g]["stratum"](d, label)].append(
                            (i, d["decision"], d["checks"], label))
        except ValueError as e:
            raise ConfigError(f"{arm}: decisions and table differ in length ({e})") from e

        wanted: dict[int, list[Item]] = collections.defaultdict(list)
        for g in groups:
            st = strata[g]
            take = allocate({k: len(v) for k, v in st.items()}, sizes[g], GROUPS[g]["alloc"])
            for k in sorted(st):
                rows = st[k]
                random.Random(f"{seed}|{g}|{k}").shuffle(rows)
                n = take.get(k, 0)
                for i, decision, checks, label in rows[:n]:
                    it = Item(group=g, stratum=k, arm=arm, index=i, label=label,
                              decision=decision, checks=checks,
                              weight=round(len(rows) / n, 6))
                    wanted[i].append(it)
                    items.append(it)
        for i, row in enumerate(iter_rows(table)):
            if i in wanted:
                if arm not in headers:
                    headers[arm] = list(row)
                for it in wanted[i]:
                    it.row = tuple(row.values())
        headers.setdefault(arm, [])

    items.sort(key=lambda it: (it.group, it.arm, it.index))
    random.Random(f"{seed}|order").shuffle(items)

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(POOL_FIELDS)
    for n, it in enumerate(items):
        w.writerow([n, it.group, it.stratum, it.arm, it.index, it.label, it.decision,
                    it.checks, it.weight])
    return items, headers, inputs, buf.getvalue()


def row_key(input_sha: str, arm: str, index: int, model: str, prompt: dict,
            endpoint: str = "", guided: bool = True) -> str:
    """Guided, this is byte for byte c2_judge's key, so a c2 run resumes here untouched."""
    return sha("|".join([input_sha, arm, str(index), model, prompt["prompt_sha"],
                         prompt["schema_sha"] if guided else "guided=off", endpoint]))[:32]


# --------------------------------------------------------------------------------------
# results and provenance files


def read_results(path: str) -> tuple[dict[str, dict], int, int]:
    """-> (latest record per key, lines read, unreadable lines)."""
    latest: dict[str, dict] = {}
    n = bad = 0
    if not os.path.exists(path):
        return latest, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1                   # a line torn by a hard kill
                continue
            latest[rec["key"]] = rec
            n += 1
    return latest, n, bad


PINNED = ("prompt", "prompt_sha", "schema_sha", "key_order", "guided", "decoding", "pool",
          "sizes", "pool_seed", "pool_sha", "inputs", "mode")
# Keys a run.json written by c2_judge.py cannot have; its runs were all guided.
PINNED_DEFAULTS = {"guided": True}


def check_run_json(out: str, meta: dict, session: dict) -> None:
    path = os.path.join(out, "run.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        diff = [k for k in PINNED if old.get(k, PINNED_DEFAULTS.get(k)) != meta.get(k)]
        if diff:
            raise ConfigError(
                f"{rel(out)} was started with different {diff}. Use a new --run-tag; "
                f"never pool results across configurations.")
        old.setdefault("sessions", []).append(session)
        write_json(path, old)
    else:
        write_json(path, {**meta, "created": now_iso(), "sessions": [session]})


def summarise(results_path: str, meta: dict) -> dict:
    latest, lines, bad = read_results(results_path)
    recs = list(latest.values())
    out: dict = {"prompt": meta["prompt"], "pool": meta["pool"], "mode": meta["mode"],
                 "rows": len(recs), "lines": lines, "unreadable_lines": bad,
                 "parse_errors": sum(bool(r.get("parse_error")) for r in recs),
                 "errors": dict(collections.Counter(
                     r["error"]["class"] for r in recs if r.get("error"))),
                 "key_order_violations": sum(r.get("key_order_ok") is False for r in recs),
                 "guided": meta.get("guided", True),
                 "replies_with_reasoning": sum(bool(r.get("reasoning")) for r in recs),
                 "parse_notes": dict(collections.Counter(
                     r["parse_note"] for r in recs if r.get("parse_note"))),
                 "by_group": {}, "by_endpoint": {}}
    for g in sorted({r["group"] for r in recs}):
        rs = [r for r in recs if r["group"] == g]
        judged = [r for r in rs if r.get("verdict")]
        wsum = sum(r["weight"] for r in judged) or 1.0

        def share(verdicts, weighted):
            hit = [r for r in judged if r["verdict"] in verdicts]
            if weighted:
                return round(sum(r["weight"] for r in hit) / wsum, 4)
            return round(len(hit) / max(1, len(judged)), 4)

        out["by_group"][g] = {
            "rows": len(rs), "judged": len(judged),
            "verdicts": dict(collections.Counter(r["verdict"] for r in judged)),
            "not_judged": len(rs) - len(judged),
            "inconsistent_rate": share({"inconsistent"}, False),
            "inconsistent_or_unsure_rate": share({"inconsistent", "unsure"}, False),
            "inconsistent_rate_weighted": share({"inconsistent"}, True),
            "by_stratum": {s: dict(collections.Counter(r.get("verdict") or "not_judged"
                                                       for r in rs if r["stratum"] == s))
                           for s in sorted({r["stratum"] for r in rs})},
        }
    for e in sorted({r["endpoint"] for r in recs}):
        rs = [r for r in recs if r["endpoint"] == e]
        lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
        ctok = [r["completion_tokens"] for r in rs if r.get("completion_tokens") is not None]
        ptok = [r["prompt_tokens"] for r in rs if r.get("prompt_tokens") is not None]
        out["by_endpoint"][e] = {
            "rows": len(rs), "model": sorted({r["model"] for r in rs}),
            "latency_s_p50": quantile(lat, 0.5), "latency_s_p90": quantile(lat, 0.9),
            "latency_s_max": max(lat) if lat else None,
            "completion_tokens_p50": quantile(ctok, 0.5),
            "completion_tokens_max": max(ctok) if ctok else None,
            "prompt_tokens_p50": quantile(ptok, 0.5),
            "truncated": sum((r.get("error") or {}).get("class") == "length" for r in rs)}
    if meta["mode"] == "compare":
        by_row: dict[tuple, dict[str, str]] = collections.defaultdict(dict)
        for r in recs:
            if r.get("verdict"):
                by_row[(r["arm"], r["index"])][r["endpoint"]] = r["verdict"]
        eps = sorted(out["by_endpoint"])
        pairs = {}
        for a_i, a in enumerate(eps):
            for b in eps[a_i + 1:]:
                both = [v for v in by_row.values() if a in v and b in v]
                pairs[f"{a}|{b}"] = {
                    "rows": len(both), "agree": sum(v[a] == v[b] for v in both),
                    "table": dict(collections.Counter(f"{v[a]}|{v[b]}" for v in both))}
        out["agreement"] = pairs
    return out


# --------------------------------------------------------------------------------------
# request quota, per ngrok account


class Quota:
    """Requests per account per UTC calendar month, shared by every run through a file."""

    def __init__(self, path: str):
        self.path = path
        self.pending: collections.Counter = collections.Counter()
        self.base = self._read()

    def _read(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except ValueError as e:
            raise ConfigError(f"{self.path} is not valid JSON") from e

    @staticmethod
    def _key(account: str) -> str:
        return f"{account}/{dt.datetime.now(dt.timezone.utc):%Y-%m}"

    def used(self, account: str) -> int:
        k = self._key(account)
        return self.base.get(k, 0) + self.pending[k]

    def add(self, account: str) -> None:
        self.pending[self._key(account)] += 1
        if sum(self.pending.values()) >= 50:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            text = f.read()
            data = json.loads(text) if text.strip() else {}
            for k, v in self.pending.items():
                data[k] = data.get(k, 0) + v
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)
        self.base = data
        self.pending.clear()


# --------------------------------------------------------------------------------------
# concurrency window


class Tuner:
    """In-flight window for one endpoint.

    Measured in rounds of max(8, limit) completions. While growing, each round that beat
    the previous one by 3% or more raises the limit by 25%; a round that did not steps
    back to the previous limit and holds. Holding re-tries growth every 10 rounds.
    Rounds where the queue ran dry are ignored, since the limit was not what bounded
    them. 429s and timeouts halve the limit, at most once per 5 s.
    """

    def __init__(self, start: int, lo: int, hi: int):
        self.start, self.lo, self.hi = start, lo, hi
        self.last_cut = 0.0
        self.reset()

    def reset(self) -> None:
        self.limit = self.start
        self.mode, self.prev, self.prev_limit, self.hold = "grow", None, None, 0
        self.saturated = False
        self._round()

    def _round(self) -> None:
        self.t0, self.n, self.lat = time.monotonic(), 0, []

    def on_success(self, latency: float, binding: bool, timeout: float) -> None:
        self.n += 1
        self.lat.append(latency)
        if self.n < max(8, self.limit):
            return
        if not binding:
            return self._round()
        tput = self.n / max(1e-6, time.monotonic() - self.t0)
        slow = (quantile(self.lat, 0.9) or 0) > 0.5 * timeout
        if self.mode == "grow":
            if self.prev is not None and tput < self.prev * 1.03:
                self.limit, self.mode, self.hold = self.prev_limit, "hold", 0
            elif slow or self.saturated or self.limit >= self.hi:
                self.mode, self.hold = "hold", 0
            else:
                self.prev, self.prev_limit = tput, self.limit
                self.limit = min(self.hi, self.limit + max(2, self.limit // 4))
        else:
            self.hold += 1
            if self.hold >= 10 and not slow and not self.saturated:
                self.mode, self.prev = "grow", None
        self._round()

    def on_overload(self) -> None:
        now = time.monotonic()
        if now - self.last_cut < 5:
            return
        self.last_cut = now
        self.limit = max(self.lo, self.limit // 2)
        self.mode, self.hold, self.prev = "hold", 0, None
        self._round()


# --------------------------------------------------------------------------------------
# endpoint


class QuotaHit(Exception):
    pass


def classify(status: int, ngrok_code: str | None) -> str:
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "overload"
    if status == 524:                  # Cloudflare: the origin sent nothing for ~100 s
        return "timeout"
    if ngrok_code or status in (502, 503, 504, 520, 521, 522, 523, 525, 526, 527, 530):
        return "down"                  # 530 = Cloudflare tunnel not connected
    if status in (400, 404, 405, 413, 422):
        return "bad_request"
    return "retry"


class Endpoint:
    def __init__(self, cfg: dict, run: Run):
        self.cfg, self.run = cfg, run
        self.name = cfg["name"]
        self.model = cfg["model"]
        self.account = cfg.get("account") or self.name
        self.quota_limit = int(cfg.get("quota", 0))
        self.extra = cfg.get("extra") or {}
        self.metrics = bool(cfg.get("metrics", False))
        hi = int(cfg.get("max_concurrency", 128))
        self.tuner = Tuner(int(cfg.get("concurrency", 8)), int(cfg.get("min_concurrency", 1)), hi)
        self.max_connections = hi + 8
        self.state = "starting"
        self.inflight = 0
        self.client: httpx.AsyncClient | None = None
        self.queue: collections.deque = collections.deque()
        self.stats: collections.Counter = collections.Counter()
        self.errors: collections.Counter = collections.Counter()
        self.events: collections.deque = collections.deque(maxlen=30)
        self.window: collections.deque = collections.deque()
        self.fail_streak = self.bad_streak = 0
        self.completions = self.parse_failures = 0
        self.served: list[str] = []
        self.canary: dict | None = None
        self.down_since: float | None = None
        self.probe_task: asyncio.Task | None = None
        self.root = self.api = None
        self.headers = dict(HEADERS)
        try:
            self.load_env()
        except ConfigError:
            pass                       # no URL yet: preflight reports it, the probe waits

    # -- configuration ---------------------------------------------------------------
    def load_env(self) -> None:
        env_file = resolve(self.cfg.get("env_file", ".env"))
        try:
            env = read_env(env_file)
        except FileNotFoundError as e:
            raise ConfigError(f"endpoint {self.name}: no env file {env_file}") from e
        url = env.get(self.cfg["url_key"])
        if not url:
            raise ConfigError(f"endpoint {self.name}: {self.cfg['url_key']} not in {env_file}")
        url = (url if "://" in url else "https://" + url).rstrip("/")
        root = url[:-3] if url.endswith("/v1") else url
        headers = dict(HEADERS)
        if self.cfg.get("auth_key"):
            cred = env.get(self.cfg["auth_key"])
            if not cred:
                raise ConfigError(f"endpoint {self.name}: {self.cfg['auth_key']} not in "
                                  f"{env_file}")
            if self.cfg.get("auth_scheme", "bearer") == "basic":
                headers["Authorization"] = "Basic " + base64.b64encode(cred.encode()).decode()
            else:
                headers["Authorization"] = f"Bearer {cred}"
        self.root, self.api, self.headers = root, root + "/v1", headers

    def event(self, text: str, loud: bool = True) -> None:
        self.events.append(f"{now_iso()} {text}")
        if loud:
            log(f"{self.name}: {text}")

    # -- transport -------------------------------------------------------------------
    async def call(self, method: str, path: str, body: bytes | None, timeout: float
                   ) -> tuple[str, object]:
        """-> ("ok", response) or (failure class, detail). Never raises for the network."""
        if self.api is None:
            return "down", f"no URL: {self.cfg['url_key']} is not in the env file"
        if self.quota_limit and self.run.quota.used(self.account) >= self.quota_limit:
            return "quota", f"account {self.account} reached {self.quota_limit} requests"
        self.run.quota.add(self.account)
        self.stats["requests"] += 1
        url = self.root + path if path == "/metrics" else self.api + path
        try:
            r = await self.client.request(
                method, url, content=body, headers=self.headers,
                timeout=httpx.Timeout(timeout, connect=min(15.0, timeout)))
        except httpx.ConnectTimeout:
            return "down", "connect timeout"
        except httpx.TimeoutException as e:
            return "timeout", type(e).__name__
        except httpx.TransportError as e:
            return "down", f"{type(e).__name__}: {str(e)[:160]}"
        if r.status_code == 200:
            return "ok", r
        text = r.text[:400]
        m = NGROK_CODE.search(text)
        code = r.headers.get("ngrok-error-code") or (m.group(0) if m else None)
        kind = classify(r.status_code, code)
        return kind, f"HTTP {r.status_code}{' ' + code if code else ''}: {text[:200]}"

    async def poll_metrics(self) -> None:
        while not self.run.stopping:
            await asyncio.sleep(10)
            if self.state != "up":
                continue
            kind, r = await self.call("GET", "/metrics", None, 10)
            if kind == "ok":
                waiting = sum(float(v) for v in WAITING.findall(r.text))
                self.tuner.saturated = waiting > 0

    # -- health ----------------------------------------------------------------------
    async def preflight(self) -> str:
        """/v1/models lists the model; then a canary. Guided, it proves the schema is
        enforced; unguided there is nothing to enforce, so it only has to parse."""
        if self.api is None:
            try:
                self.load_env()
            except ConfigError as e:
                return self.set_down(f"waiting for its URL ({e})")
        kind, r = await self.call("GET", "/models", None, 30)
        if kind == "auth":
            self.reload_env()
            kind, r = await self.call("GET", "/models", None, 30)
        if kind != "ok":
            return self.set_down(f"/v1/models: {r}")
        try:
            self.served = [m["id"] for m in r.json()["data"]]
        except (ValueError, KeyError, TypeError):
            return self.set_down("/v1/models: unexpected payload")
        if self.model not in self.served:
            return self.refuse(f"model {self.model!r} is not served; server has {self.served}")

        run = self.run
        order = run.prompt["order"]
        payload = build_payload(
            self.model, run.prompt, run.render(run.canary_item), run.decoding, self.extra,
            schema=build_schema(order, (CANARY,)) if run.guided else None,
            name="c2_canary", guided=run.guided)
        kind, r = await self.call("POST", "/chat/completions", json.dumps(payload).encode(),
                                  run.args.timeout)
        if kind == "auth":
            self.reload_env()
            kind, r = await self.call("POST", "/chat/completions",
                                      json.dumps(payload).encode(), run.args.timeout)
        if kind == "bad_request":
            return self.refuse(f"canary request rejected: {r}")
        if kind != "ok":
            return self.set_down(f"canary: {r}")
        try:
            data = r.json()
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError):
            return self.set_down("canary: malformed chat response")
        finish = choice.get("finish_reason")
        ctok = (data.get("usage") or {}).get("completion_tokens")
        self.canary = {"at": now_iso(), "guided": run.guided, "finish_reason": finish,
                       "completion_tokens": ctok, "content": content[:300]}
        if finish == "length":
            self.event(f"WARNING canary hit max_tokens={run.decoding['max_tokens']}; the "
                       f"canary check is inconclusive, parse-error breaker still armed")
        elif run.guided:
            try:
                obj = json.loads(content)
            except ValueError:
                return self.refuse(f"canary reply is not JSON, so response_format is not "
                                   f"enforced: {content[:120]!r}")
            if not isinstance(obj, dict) or obj.get("verdict") != CANARY:
                return self.refuse(f"server ignores response_format: canary verdict "
                                   f"{obj.get('verdict') if isinstance(obj, dict) else obj!r}")
            if tuple(obj) != tuple(order):
                return self.refuse(f"server does not enforce key order: got {list(obj)}, "
                                   f"schema {list(order)}")
        else:
            # Nothing to enforce, and one row is too little to judge a model on, so a canary
            # that does not parse is a warning; the parse-error breaker in handle() decides.
            parsed = parse_tolerant(content, order)
            self.canary["parsed"] = {k: v for k, v in parsed.items() if k != "reasoning"}
            if parsed.get("parse_error"):
                self.event(f"WARNING unguided canary did not parse ({parsed['parse_error']}); "
                           f"parse-error breaker armed")
        return self.set_up()

    def reload_env(self) -> None:
        try:
            self.load_env()
            self.event("re-read env after 401/403", loud=False)
        except ConfigError as e:
            self.event(f"re-reading env failed: {e}")

    def set_up(self) -> str:
        was = self.down_since
        self.state, self.down_since = "up", None
        self.fail_streak = 0
        self.tuner.reset()
        self.event(f"up, model {self.model}, window {self.tuner.limit}"
                   + (f" (after {fmt_dur(time.monotonic() - was)} down)" if was else ""))
        return "up"

    def set_down(self, why: str) -> str:
        if self.state == "up" or self.down_since is None:
            self.down_since = time.monotonic()
        if self.state != "down":
            self.event(f"DOWN: {why}")
        self.state = "down"
        return "down"

    def refuse(self, why: str) -> str:
        self.state = "refused"
        self.event(f"REFUSED: {why}")
        return "refused"

    def mark_down(self, why: str) -> None:
        if self.state != "up":
            return
        self.set_down(why)
        self.start_probe()

    def start_probe(self) -> None:
        if self.probe_task is None or self.probe_task.done():
            self.probe_task = asyncio.create_task(self.probe_loop())

    async def probe_loop(self) -> None:
        """Re-read the env file every few seconds (free); probe the network with backoff,
        or at once when the URL in the env file has changed."""
        args = self.run.args
        delay, last, seen = args.probe_min, time.monotonic(), self.api
        while not self.run.stopping and self.state == "down":
            await asyncio.sleep(min(5.0, args.probe_min) * random.uniform(0.8, 1.2))
            if self.run.stopping:
                return
            try:
                self.load_env()
            except ConfigError:
                continue                           # URL still not there; nothing to probe
            changed = self.api != seen
            if changed:
                self.event(f"{self.cfg['url_key']} changed in the env file; probing now")
                seen, delay = self.api, args.probe_min
            elif time.monotonic() - last < delay:
                continue
            if self.quota_limit and self.run.quota.used(self.account) >= self.quota_limit:
                self.state = "quota"
                self.event(f"QUOTA: account {self.account} reached {self.quota_limit}")
                return
            last = time.monotonic()
            if await self.preflight() != "down":
                return
            if not changed:
                delay = min(args.probe_max, delay * 2)


# --------------------------------------------------------------------------------------
# run


class Run:
    def __init__(self, args, prompt: dict, decoding: dict, meta: dict, out: str,
                 items_by_ep: dict[str, list[Item]], shared: bool, headers: dict,
                 endpoint_cfgs: list[dict], quota: Quota, done: set[str],
                 canary_item: Item):
        self.args, self.prompt, self.decoding, self.meta, self.out = (
            args, prompt, decoding, meta, out)
        self.headers = headers
        self.guided = args.guided == "on"
        self.quota = quota
        self.done = done
        self.canary_item = canary_item
        self.shared = shared
        self.session = now_iso()
        self.stopping = False
        self.stop_reason = ""
        self.finished: asyncio.Event | None = None
        self.sleeping = 0
        self.lines = 0
        self.written = 0
        self.duplicates = 0
        self.endpoints = [Endpoint(c, self) for c in endpoint_cfgs]
        if shared:
            q = collections.deque(items_by_ep["*"])
            for ep in self.endpoints:
                ep.queue = q
        else:
            for ep in self.endpoints:
                ep.queue = collections.deque(items_by_ep[ep.name])
        self.planned: set[str] = set()                 # every key this session should end with
        self.results_path = os.path.join(out, "results.jsonl")
        self.out_f = None

    # -- rendering and records -------------------------------------------------------
    def render(self, it: Item) -> str:
        return render_user(self.prompt["user"], dict(zip(self.headers[it.arm], it.row)))

    def queues(self) -> list[collections.deque]:
        seen, out = set(), []
        for ep in self.endpoints:
            if id(ep.queue) not in seen and (self.shared or ep.state not in ("refused", "quota")):
                seen.add(id(ep.queue))
                out.append(ep.queue)
        return out

    def remaining(self) -> int:
        return (sum(len(q) for q in self.queues()) + sum(ep.inflight for ep in self.endpoints)
                + self.sleeping)

    def record(self, ep: Endpoint, it: Item, content: str | None, info: dict,
               parsed: dict | None = None, error: tuple[str, str] | None = None) -> None:
        if it.key in self.done:
            self.duplicates += 1
            return
        parsed = parsed or {}
        rec = {
            "key": it.key, "run_tag": self.args.run_tag, "session": self.session,
            "group": it.group, "stratum": it.stratum, "arm": it.arm, "index": it.index,
            "label": it.label, "rules_decision": it.decision, "rules_checks": it.checks,
            "weight": it.weight,
            "endpoint": ep.name, "model": ep.model, "prompt": self.prompt["name"],
            "prompt_sha": self.prompt["prompt_sha"], "schema_sha": self.prompt["schema_sha"],
            "guided": self.guided,
            "decoding": self.decoding, "extra": ep.extra or None,
            "input_sha": self.meta["inputs"][it.arm]["table_sha"],
            "verdict": parsed.get("verdict"), "short_reason": parsed.get("short_reason"),
            "reasoning": parsed.get("reasoning"), "parse_note": parsed.get("parse_note"),
            "key_order_ok": parsed.get("key_order_ok"),
            "parse_error": parsed.get("parse_error"),
            "error": {"class": error[0], "detail": error[1]} if error else None,
            "attempts": it.attempts + it.parse_retries + 1,
            "raw": content, **info, "ts": now_iso(),
        }
        self.out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.out_f.flush()
        self.lines += 1
        if self.lines % 50 == 0:
            os.fsync(self.out_f.fileno())
        self.done.add(it.key)
        self.written += 1
        ep.stats["rows"] += 1
        ep.window.append(time.monotonic())
        it.row = None

    # -- one request -----------------------------------------------------------------
    async def backoff(self, it: Item, q: collections.deque, attempt: int) -> None:
        self.sleeping += 1
        try:
            await asyncio.sleep(min(self.args.backoff_max, 2 ** attempt) * random.uniform(0.5, 1.0))
        finally:
            self.sleeping -= 1
        if not self.stopping:
            q.appendleft(it)

    async def handle(self, ep: Endpoint, it: Item, q: collections.deque) -> None:
        body = json.dumps(build_payload(ep.model, self.prompt, self.render(it), self.decoding,
                                        ep.extra, guided=self.guided)).encode()
        t0 = time.monotonic()
        try:
            kind, r = await ep.call("POST", "/chat/completions", body, self.args.timeout)
        finally:
            ep.inflight -= 1
        latency = round(time.monotonic() - t0, 3)

        if kind == "ok":
            try:
                data = r.json()
                choice = data["choices"][0]
                content = choice["message"].get("content") or ""
                usage = data.get("usage") or {}
            except (ValueError, KeyError, IndexError, TypeError) as e:
                kind, r = "retry", f"malformed chat response: {e}"
            else:
                ep.fail_streak = ep.bad_streak = 0
                info = {"latency_s": latency, "finish_reason": choice.get("finish_reason"),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens")}
                if info["finish_reason"] == "length":
                    ep.errors["length"] += 1
                    self.record(ep, it, content, info,
                                error=("length", f"finish_reason=length at max_tokens="
                                                 f"{self.decoding['max_tokens']}"))
                    return
                parse = parse_strict if self.guided else parse_tolerant
                parsed = parse(content, self.prompt["order"])
                if parsed.get("parse_error"):
                    ep.errors["parse"] += 1
                    if it.parse_retries < 1:
                        it.parse_retries += 1
                        q.appendleft(it)
                        return
                ep.completions += 1
                ep.parse_failures += bool(parsed.get("parse_error"))
                self.record(ep, it, content, info, parsed=parsed)
                ep.tuner.on_success(latency, binding=bool(q), timeout=self.args.timeout)
                if ep.completions == 20 and ep.parse_failures == 20:
                    ep.refuse(f"the first 20 replies all failed "
                              f"{'strict' if self.guided else 'tolerant'} parsing")
                return

        ep.errors[kind] += 1
        if kind == "down":
            q.appendleft(it)
            ep.mark_down(str(r))
            return
        if kind == "quota":
            q.appendleft(it)
            if ep.state == "up":
                ep.state = "quota"
                ep.event(f"QUOTA: {r}")
            return
        if kind == "auth":
            q.appendleft(it)
            if not it.auth_retried:
                it.auth_retried = True
                ep.reload_env()
            else:
                ep.mark_down(f"auth still failing after re-reading env: {r}")
            return
        if kind == "bad_request":
            ep.bad_streak += 1
            self.record(ep, it, None, {"latency_s": latency}, error=("bad_request", str(r)))
            if ep.bad_streak >= 20 and ep.state == "up":
                ep.refuse(f"20 bad requests in a row, last: {r}")
            return
        # overload, timeout, other 5xx
        if kind in ("overload", "timeout"):
            ep.tuner.on_overload()
        ep.fail_streak += 1
        if ep.fail_streak >= max(5, ep.tuner.lo) and kind != "overload":
            q.appendleft(it)
            ep.mark_down(f"{ep.fail_streak} failures in a row, last {kind}: {r}")
            return
        it.attempts += 1
        if it.attempts >= self.args.max_attempts:
            self.record(ep, it, None, {"latency_s": latency}, error=(kind, str(r)))
            return
        await self.backoff(it, q, it.attempts)

    # -- loops -----------------------------------------------------------------------
    async def dispatch(self, ep: Endpoint) -> None:
        tasks: set[asyncio.Task] = set()
        while True:
            if self.stopping:
                if not tasks:
                    return
            elif ep.state == "up":
                q = ep.queue
                while q and ep.state == "up" and ep.inflight < ep.tuner.limit:
                    it = q.popleft()
                    ep.inflight += 1
                    tasks.add(asyncio.create_task(self.handle(ep, it, q)))
            if tasks:
                _, tasks = await asyncio.wait(tasks, timeout=0.5,
                                              return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.sleep(0.2)

    def stop(self, reason: str) -> None:
        if not self.stopping:
            self.stopping, self.stop_reason = True, reason
            log(f"stopping: {reason}")
        self.finished.set()

    async def monitor(self) -> None:
        tick = 0.1 if self.args.progress_every < 5 else 1.0
        last_progress = last_summary = time.monotonic()
        all_down_since = None
        while not self.stopping:
            await asyncio.sleep(tick)
            now = time.monotonic()
            if self.remaining() == 0 and all(ep.state != "starting" for ep in self.endpoints):
                self.stop("queue drained")
                break
            live = [ep for ep in self.endpoints if ep.state not in ("refused", "quota")
                    and (ep.queue or ep.inflight or self.sleeping)]
            if not live:
                states = ", ".join(f"{ep.name}={ep.state}" for ep in self.endpoints)
                self.stop(f"no usable endpoint has work left ({states})")
                break
            if any(ep.state in ("up", "starting") for ep in live):
                all_down_since = None
            else:
                all_down_since = all_down_since or now
                if now - all_down_since > self.args.give_up_after:
                    self.stop(f"every endpoint down for {fmt_dur(now - all_down_since)}")
                    break
            if now - last_progress >= self.args.progress_every:
                self.progress()
                self.write_endpoints()
                last_progress = now
            if now - last_summary >= 600:
                self.write_summary()
                last_summary = now

    def progress(self) -> None:
        now = time.monotonic()
        parts, recent = [], 0
        for ep in self.endpoints:
            while ep.window and now - ep.window[0] > 300:
                ep.window.popleft()
            recent += len(ep.window)
            per_min = sum(1 for t in ep.window if now - t <= 60)
            state = (f"down {fmt_dur(now - ep.down_since)}" if ep.state == "down" and ep.down_since
                     else ep.state)
            errs = ",".join(f"{k}:{v}" for k, v in sorted(ep.errors.items()))
            parts.append(f"{ep.name} {state} window {ep.tuner.limit} in-flight {ep.inflight} "
                         f"{per_min}/min" + (f" err[{errs}]" if errs else ""))
        done = len(self.planned & self.done)
        left = len(self.planned) - done
        rate = recent / 5 if recent else 0
        eta = fmt_dur(left / rate * 60) if rate else "?"
        log(f"{done:,}/{len(self.planned):,} rows | " + " | ".join(parts) + f" | ETA {eta}")

    def write_endpoints(self) -> None:
        self.quota.flush()
        write_json(os.path.join(self.out, "endpoints.json"), {
            "session": self.session, "updated": now_iso(), "quota_file": rel(self.quota.path),
            "endpoints": [{
                "name": ep.name, "env_file": ep.cfg.get("env_file", ".env"),
                "url_key": ep.cfg["url_key"], "auth_key": ep.cfg.get("auth_key") or None,
                "model": ep.model, "served": ep.served, "state": ep.state,
                "account": ep.account, "quota": ep.quota_limit,
                "quota_used_this_month": self.quota.used(ep.account),
                "requests_this_session": ep.stats["requests"], "rows_this_session": ep.stats["rows"],
                "window": ep.tuner.limit, "window_bounds": [ep.tuner.lo, ep.tuner.hi],
                "errors": dict(ep.errors), "extra": ep.extra or None, "canary": ep.canary,
                "events": list(ep.events)} for ep in self.endpoints]})

    def write_summary(self) -> dict:
        s = summarise(self.results_path, self.meta)
        s.update({"planned": len(self.planned), "missing": len(self.planned - self.done),
                  "stop_reason": self.stop_reason, "updated": now_iso()})
        write_json(os.path.join(self.out, "summary.json"), s)
        return s

    async def main(self) -> int:
        self.finished = asyncio.Event()
        loop = asyncio.get_running_loop()

        def on_signal():
            if self.stopping:
                log("second interrupt: exiting now")
                self.out_f.flush()
                os.fsync(self.out_f.fileno())
                self.quota.flush()
                os._exit(130)
            self.stop("interrupted; waiting for in-flight requests (interrupt again to exit)")

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, on_signal)
            except (NotImplementedError, RuntimeError, ValueError):
                pass                                   # not the main thread (tests)

        needs_nl = False
        if os.path.exists(self.results_path) and os.path.getsize(self.results_path):
            with open(self.results_path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_nl = f.read(1) != b"\n"
        self.out_f = open(self.results_path, "a", encoding="utf-8")
        if needs_nl:
            self.out_f.write("\n")                     # isolate a line torn by a hard kill

        log(f"{len(self.planned - self.done):,} rows to judge ({len(self.planned):,} planned, "
            f"{len(self.planned & self.done):,} already recorded) with "
            f"{', '.join(ep.name for ep in self.endpoints)}")
        for ep in self.endpoints:
            ep.client = httpx.AsyncClient(limits=httpx.Limits(
                max_connections=ep.max_connections,
                max_keepalive_connections=ep.max_connections))
        try:
            states = await asyncio.gather(*(ep.preflight() for ep in self.endpoints))
            for ep, st in zip(self.endpoints, states):
                if st == "down":
                    ep.start_probe()
            workers = [asyncio.create_task(self.dispatch(ep)) for ep in self.endpoints]
            extras = [asyncio.create_task(ep.poll_metrics()) for ep in self.endpoints
                      if ep.metrics]
            monitor = asyncio.create_task(self.monitor())
            await self.finished.wait()
            self.stopping = True
            await asyncio.wait(workers, timeout=self.args.timeout + 30)
            for t in [monitor, *extras, *(ep.probe_task for ep in self.endpoints
                                          if ep.probe_task)]:
                t.cancel()
        finally:
            for ep in self.endpoints:
                if ep.client:
                    await ep.client.aclose()
            self.out_f.flush()
            os.fsync(self.out_f.fileno())
            self.out_f.close()
            self.quota.flush()
            self.write_endpoints()
        s = self.write_summary()
        log(f"wrote {self.written:,} rows this session; {s['missing']:,} of {s['planned']:,} "
            f"planned rows still missing; {s['parse_errors']} parse errors, errors "
            f"{s['errors']}; duplicates suppressed {self.duplicates}")
        return EXIT_OK if s["missing"] == 0 else EXIT_INCOMPLETE


# --------------------------------------------------------------------------------------
# entry point


def load_endpoints(path: str, only: list[str]) -> tuple[list[dict], str]:
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"no endpoints file {path}") from e
    eps = [e for e in cfg.get("endpoint", []) if e.get("enabled", True)]
    if only:
        unknown = sorted(set(only) - {e["name"] for e in cfg.get("endpoint", [])})
        if unknown:
            raise ConfigError(f"--endpoint {unknown} not in {path}")
        eps = [e for e in cfg.get("endpoint", []) if e["name"] in only]
    if not eps:
        raise ConfigError(f"no enabled endpoint in {path}")
    for e in eps:
        missing = [k for k in ("name", "url_key", "model") if not e.get(k)]
        if missing:
            raise ConfigError(f"{path}: endpoint {e.get('name')!r} lacks {missing}")
    return eps, resolve(cfg.get("quota_file", "runs/c2_quota.json"))


def parse_sizes(spec: str) -> dict[str, int]:
    out = {}
    for part in spec.split(","):
        g, _, n = part.partition("=")
        if g.strip() not in GROUPS:
            raise ConfigError(f"--sizes group {g!r} not in {sorted(GROUPS)}")
        out[g.strip()] = int(n)
    return out


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompt", default="c2", help="directory under prompts/, or a path")
    ap.add_argument("--pool", choices=sorted(SIZES), default="pilot")
    ap.add_argument("--sizes", default="", help="override pool sizes: kept=50,positive=25,test=25"
                                                " (0 = every eligible row)")
    ap.add_argument("--seed", type=int, default=42, help="pool sampling and decoding seed")
    ap.add_argument("--run-dir", default="runs/split-ctgan-e300-gpu")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--out", default="", help="default <run-dir>/c2/<run-tag>")
    ap.add_argument("--endpoints", default="configs/endpoints.toml")
    ap.add_argument("--endpoint", action="append", default=[],
                    help="use only this endpoint (repeatable)")
    ap.add_argument("--mode", choices=["shard", "compare"], default="shard")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows of the pool order")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-send rows whose latest record is an error or parse error")
    ap.add_argument("--guided", choices=["on", "off"], default="on",
                    help="on: response_format carries the two-key schema and the reply must be "
                         "exactly that object. off: no schema, so the prompt may reason in "
                         "plain text and the last JSON object with a verdict is taken")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default 96 with --guided on, 1024 with --guided off")
    ap.add_argument("--timeout", type=float, default=95.0,
                    help="seconds per request; under Cloudflare's ~100 s 524 cut-off")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--backoff-max", type=float, default=60.0)
    ap.add_argument("--probe-min", type=float, default=15.0)
    ap.add_argument("--probe-max", type=float, default=300.0)
    ap.add_argument("--give-up-after", type=float, default=3600.0,
                    help="seconds with every endpoint down before stopping")
    ap.add_argument("--progress-every", type=float, default=60.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="write pool, prompt and request examples; call nothing")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_main(args)
    except ConfigError as e:
        print(f"llmjudge: refused: {e}", file=sys.stderr, flush=True)
        return EXIT_CONFIG


def run_main(args: argparse.Namespace) -> int:
    prompt = load_prompt(args.prompt)
    guided = args.guided == "on"
    if args.max_tokens is None:
        args.max_tokens = 96 if guided else 1024
    sizes = dict(SIZES[args.pool])
    pool_name = args.pool
    if args.sizes:
        sizes.update(parse_sizes(args.sizes))
        pool_name = f"{args.pool}+custom"
    run_dir = resolve(args.run_dir)
    out = resolve(args.out) if args.out else os.path.join(run_dir, "c2", args.run_tag)
    decoding = {"temperature": args.temperature, "seed": args.seed,
                "max_tokens": args.max_tokens}

    items, headers, inputs, pool_csv = build_pool(run_dir, sizes, args.seed)
    if not items:
        raise ConfigError("the pool is empty")
    meta = {"prompt": prompt["name"], "prompt_dir": rel(prompt["dir"]),
            "prompt_sha": prompt["prompt_sha"], "key_order": list(prompt["order"]),
            "guided": guided,
            # Unguided nothing is sent, but the sha still pins the key order the prompt asks
            # for, and it keeps the resume key the same as c2_judge's on the guided path.
            "schema": prompt["schema"] if guided else None,
            "schema_sha": prompt["schema_sha"],
            "decoding": decoding, "pool": pool_name, "sizes": sizes, "pool_seed": args.seed,
            "pool_sha": sha(pool_csv), "run_dir": rel(run_dir), "inputs": inputs,
            "mode": args.mode}
    session = {"started": now_iso(), "argv": sys.argv[1:] if __name__ == "__main__" else None,
               "script_sha": file_sha(os.path.abspath(__file__)), "dry_run": args.dry_run,
               "pid": os.getpid()}
    os.makedirs(out, exist_ok=True)
    check_run_json(out, meta, session)
    pool_path = os.path.join(out, "pool.csv")
    with open(pool_path + ".tmp", "w", encoding="utf-8", newline="") as f:
        f.write(pool_csv)
    os.replace(pool_path + ".tmp", pool_path)

    counts = collections.Counter((it.group, it.stratum) for it in items)
    log(f"pool {pool_name}: {len(items):,} rows -> {rel(pool_path)}")
    for (g, s), n in sorted(counts.items()):
        log(f"  {g:<9} {s:<28} {n:>6}")
    log(f"prompt {prompt['name']}: key order {list(prompt['order'])}, guided decoding "
        f"{'on' if guided else 'off'}, max_tokens {args.max_tokens}, prompt sha "
        f"{prompt['prompt_sha'][:16]}, schema sha {prompt['schema_sha'][:16]}")

    example = items[0]
    user_text = render_user(prompt["user"], dict(zip(headers[example.arm], example.row)))
    with open(os.path.join(out, "prompt_example.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== SYSTEM ===\n{prompt['system']}\n\n=== USER ===\n{user_text}\n")
    write_json(os.path.join(out, "request_example.json"),
               build_payload("<model from the endpoints file>", prompt, user_text, decoding,
                             guided=guided))
    if args.dry_run:
        log(f"dry run: no calls. Wrote pool.csv, run.json, prompt_example.txt, "
            f"request_example.json in {rel(out)}")
        return EXIT_OK

    ep_cfgs, quota_file = load_endpoints(resolve(args.endpoints), args.endpoint)
    if args.mode == "shard" and len({e["model"] for e in ep_cfgs}) > 1:
        raise ConfigError(f"shard mode needs one model on every endpoint, got "
                          f"{sorted({e['model'] for e in ep_cfgs})}; use --mode compare")

    latest, lines, torn = read_results(os.path.join(out, "results.jsonl"))
    if torn:
        log(f"results.jsonl: {torn} unreadable line(s) ignored; those rows will be re-sent")
    skip = {k for k, r in latest.items()
            if not (args.retry_errors and (r.get("error") or r.get("parse_error")))}
    selected = items[:args.limit] if args.limit else items

    def key_for(it: Item, model: str, endpoint: str = "") -> str:
        return row_key(inputs[it.arm]["table_sha"], it.arm, it.index, model, prompt, endpoint,
                       guided)

    if args.mode == "shard":
        model = ep_cfgs[0]["model"]
        planned = [it.clone(key_for(it, model)) for it in selected]
        by_ep = {"*": [it for it in planned if it.key not in skip]}
        planned_keys = {it.key for it in planned}
    else:
        by_ep, planned_keys = {}, set()
        for e in ep_cfgs:
            its = [it.clone(key_for(it, e["model"], e["name"])) for it in selected]
            planned_keys |= {it.key for it in its}
            by_ep[e["name"]] = [it for it in its if it.key not in skip]

    canary = next((it for it in items if it.group == "test"), items[0])
    run = Run(args, prompt, decoding, meta, out, by_ep, args.mode == "shard", headers,
              ep_cfgs, Quota(quota_file), done=skip & planned_keys, canary_item=canary)
    run.planned = planned_keys

    pid_path = os.path.join(out, "judge.pid")
    with open(pid_path, "w") as f:
        f.write(f"{os.getpid()}\n")
    try:
        return asyncio.run(run.main())
    finally:
        with open(pid_path) as f:
            if f.read().strip() == str(os.getpid()):
                os.remove(pid_path)


if __name__ == "__main__":
    raise SystemExit(main())
