#!/usr/bin/env python3
"""Judge rows with an LLM over an OpenAI-compatible API.

    python3 -m llmjudge --items items.jsonl --out out/pilot --prompt c3 \\
        --run-tag pilot-01 --dry-run              # render the prompt, send nothing
    python3 -m llmjudge --items items.jsonl --out out/pilot --prompt c3 \\
        --run-tag pilot-01
    nohup python3 -m llmjudge --items items.jsonl --out out/c3r --prompt c3-reasoning \\
        --guided off --run-tag c3r-01 >> out/c3r.log 2>&1 &

One row per request, temperature 0. The row is rendered into the prompt's `user.md`. The
model never sees the row's id, group or stratum.

Items (`--items`, one JSON object per line)

    {"id": "ctgan_split:41772", "fields": {"age": "[70-80)", "diag_1": "250.83"}}

    id       required, unique. The only thing tying a verdict back to a row, and this
             script never parses it.
    fields   required. Filled into the `{column}` placeholders of the user template. A
             template naming a field an item lacks is refused before anything is sent.
    group    optional, opaque. It buckets `summary.json`; nothing here reads its meaning.
    stratum  optional, opaque. The same.
    weight   optional, default 1.0. It weights the rates in the summary, so a caller that
             sampled strata unequally can still report a population rate.

    Rows are sent in one seeded shuffled order, so any prefix of a run is a random sample
    of the items file. Building that file is the caller's job: stratifying a pool needs
    to know what a rule decided about each row, and this repository holds no rules.

Guided decoding
    --guided on, the default, sends the answer's JSON schema as `response_format`. The
    reply must be exactly that object, in the key order the prompt's example shows.

    --guided off sends no schema, so a prompt may reason in plain text before it answers
    (`prompts/c3-reasoning`). The verdict is then the last JSON object in the reply that
    carries the label key, so prose and code fences around it are fine, and the text
    before it is kept as `reasoning`.

    Unguided replies are long, so --max-tokens defaults to 1024 rather than 96. A reply
    cut off at the budget is recorded as an error, never as a verdict. After a pilot,
    read `truncated` in summary.json: if it is not 0, raise --max-tokens and re-run with
    --retry-errors.

Streaming (--stream)
    Cloudflare cuts a request whose origin has sent nothing for about 100 s (524), so a
    prompt that reasons before it answers can lose a row the model is still working on.
    Streamed, --timeout measures silence rather than the whole reply and may go past the
    tunnel's limit. The answer, its token counts and its finish reason are the same
    either way. Off by default; `stream = true` turns it on for one endpoint.

Throughput
    Nothing is sent in batches. Each endpoint keeps a window of in-flight requests and
    starts the next one the moment any finishes, so a 100 s row holds one slot while the
    rest keep flowing. vLLM batches whatever is in flight on its side.

    The window adapts. It grows by 25% while measured throughput keeps rising, steps back
    when a step bought nothing, and halves on 429s and timeouts. The bounds come from
    `configs/endpoints.toml`. With `metrics = true` it also stops growing while vLLM
    reports queued requests; that is off by default, because every poll is one more
    request through the tunnel.

Failures
    endpoint down   connection errors, ngrok errors (ERR_NGROK_*), 502/503/504 and
                    Cloudflare 520-523/525-527/530. The row goes back on the queue and the
                    endpoint pauses. `/v1/models` and a canary are then probed with
                    backoff (--probe-min .. --probe-max) until it comes back on its own.
                    In shard mode the other endpoints drain the queue meanwhile. Five
                    timeouts or 5xx in a row also pause it, because a dead engine can
                    still answer `/v1/models`.
    new URL         while an endpoint is down its env file is re-read every few seconds,
                    and a changed URL is probed at once. So a Colab restart, which gets a
                    new trycloudflare hostname, costs one edit to .env and the running
                    judge follows. An endpoint whose URL key is not in .env yet waits the
                    same way, so a second notebook can join mid-run -- as long as its
                    entry was in the endpoints file when the run started.
    401 / 403       the env file is re-read, since keys rotate, and the row is retried
                    once.
    429, timeout    also Cloudflare 524. The window halves and the row is retried with
                    jittered backoff, at most --max-attempts times, then recorded as an
                    error.
    400 / 422       recorded as an error for that row. Twenty in a row refuse the
                    endpoint.
    bad reply       non-JSON, or the wrong keys: retried once, then recorded with
                    parse_error. finish_reason=length is recorded as an error, never as a
                    verdict.
    ceiling         --max-requests N stops the whole run at N requests -- rows, preflight
                    and probes alike. It is what keeps a typo from judging 55,000 rows
                    against a paid API. It stops the run the way Ctrl-C does, so the
                    answers already on disk stay and a re-run resumes from them.
    gives up        only when every endpoint with work left has been down for
                    --give-up-after seconds, or none is usable.

Preflight, per endpoint, before any row
    The model listing must name the configured model, and a canary request must come back
    sound. --skip-model-check is for an API that lists no models, or that lists deployment
    names rather than the name you send; the canary then proves the endpoint alone.

    Guided, the canary's schema allows one value for one key -- the label, or the first
    key when the prompt has no label -- and the reply must be exactly that, in the
    schema's key order. So a server that silently ignores `response_format` is refused
    before a single row is sent. Unguided there is nothing to enforce: the canary only has
    to parse, and one that does not is a warning. The breaker that refuses an endpoint
    whose first 20 replies all fail parsing catches the rest.

Resume
    `<out>/results.jsonl` is append-only, one line per final answer, flushed per line and
    fsync'd every 50. A restart skips every row already recorded. `--retry-errors`
    re-sends the rows whose latest record is an error or a parse error.

    The resume key is (item id, model, prompt sha, schema sha or "guided=off"), plus the
    endpoint in compare mode. The items file's own sha is not part of it, so a pool that
    grew is judged where it grew: add rows to the file, re-run into the same directory,
    and only the new ids are sent.

    The first real run writes `<out>/run.json`, which pins the prompt, schema, sampling,
    order seed and mode. A restart with any of them changed is refused. --max-tokens is
    the exception: a reply cut off at the budget is an error and never a verdict, so
    raising it and re-running with --retry-errors recovers exactly those rows and costs
    nothing already judged. `--dry-run` writes no run.json, so one prompt after another
    can be rendered into the same directory.

    Ctrl-C once: stop sending, let the in-flight requests finish, write the summary.
    Twice: exit now. Every recorded line is already on disk either way.

Modes
    shard     one shared queue, every endpoint pulls from it. All must serve one model.
    compare   every row to every endpoint, e.g. MedGemma against a general model. With
              two replicas of one model, `--mode compare --limit 200` is the agreement
              check.

Endpoints
    One server needs no files. The key goes in --api-key or LLMJUDGE_API_KEY:

        --base-url https://host/v1 --model NAME

    Several, or one with a concurrency window of its own, go in an endpoints file:

        --endpoints configs/endpoints.toml

    Each entry there names the environment variables holding its URL and key, never the
    values. Those are read from the endpoint's env file if it has one, and from the
    process environment otherwise.

    APIs differ in the request fields they accept. `--drop seed` removes one and `--extra
    '{"max_completion_tokens": 4096}'` adds one, so renaming a field is a drop plus an
    add. Both show up in `request_example.json` under --dry-run, before anything is sent.

    They differ in transport too. A base URL that already carries a path is used exactly
    as given -- Azure's /openai/deployments/<name>?api-version=..., a gateway's prefix --
    and only a bare host gets /v1 appended. `--chat-path` moves the request,
    `--auth-header api-key` puts the key in another header (raw, with no Bearer),
    `--header 'Name: value'` adds a header to every request and an empty value removes
    one, and `--skip-model-check` is for an API that lists no models. Azure needs four of
    them at once:

    python3 -m llmjudge --skip-model-check --auth-header api-key --drop seed \\
        --base-url 'https://R.openai.azure.com/openai/deployments/D?api-version=2024-06-01' \\
        --model gpt-4o-mini ...

    The body is still OpenAI's: a messages array with the system prompt inside it, and
    response_format for guided decoding. An API with a body of its own -- Anthropic's
    native /v1/messages, where the system prompt is a top-level field and there is no
    response_format -- needs an adapter, not these flags.

Outputs in `<out>`: order.csv, run.json, results.jsonl, summary.json, endpoints.json,
prompt_example.txt, request_example.json, judge.pid.
Exit codes: 0 every planned row recorded, 2 configuration refused, 3 stopped incomplete.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import random
import re
import signal
import sys
import threading
import time
import tomllib
import urllib.parse

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "prompts")        # ships with the package, so pip install works

from .template import PLACEHOLDER, render_user

CANARY = "canary-ok"
ANSWER_FENCE = "```answer"          # marks the contract among a prompt's other examples
HEADERS = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "1"}
NGROK_CODE = re.compile(r"ERR_NGROK_\d+")
FIRST = re.compile(r'write "(\w+)" FIRST')
WAITING = re.compile(r"^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.M)
THOUGHT = re.compile(r"<unused94>.*?(?:<unused95>|$)", re.S)   # MedGemma's thinking span

ORDER_FIELDS = ["order", "id", "group", "stratum", "weight"]
ENDPOINTS_DEFAULT = "configs/endpoints.toml"

EXIT_OK, EXIT_CONFIG, EXIT_INCOMPLETE = 0, 2, 3


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
    """An absolute path. A relative one is relative to the working directory, like every
    other CLI -- not to the package, which after a pip install is site-packages."""
    return os.path.abspath(os.path.expanduser(path))


def rel(path: str) -> str:
    """A short path for the log and for run.json: relative to the working directory when
    the path is below it, absolute otherwise. `../../../../mnt/x` helps nobody."""
    r = os.path.relpath(path, os.getcwd())
    return path if r.startswith("..") else r


def join(base: str, path: str) -> str:
    """base + path, with any query string kept at the end where it belongs.

        join("https://r.openai.azure.com/openai/deployments/d?api-version=2024-06-01",
             "/chat/completions")
        -> ".../deployments/d/chat/completions?api-version=2024-06-01"
    """
    head, sep, query = base.partition("?")
    return head.rstrip("/") + path + sep + query


def parse_headers(pairs: list[str]) -> dict[str, str]:
    """`--header 'anthropic-version: 2023-06-01'` -> one entry. An empty value removes a
    header the judge would otherwise send, the way a null in `--extra` removes a field."""
    out: dict[str, str] = {}
    for p in pairs:
        name, sep, value = p.partition(":")
        if not sep or not name.strip():
            raise ConfigError(f"--header wants 'Name: value', got {p!r}")
        out[name.strip()] = value.strip()
    return out


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


def build_prompt(name: str, system: str, user: str, directory: str | None = None) -> dict:
    """The prompt as the run uses it. The text is what counts: `prompt_sha` is taken from
    it, so the same two strings resume the same run whether they came from the package, a
    folder on Drive, or a cell in a notebook."""
    system, user = system.strip(), user.strip()
    if not system or not user:
        raise ConfigError(f"prompt {name!r}: the system and user prompts cannot be empty")
    contract = read_contract(system)
    schema = contract["schema"]
    return {"name": name, "dir": directory, "system": system, "user": user,
            "contract": contract, "order": contract["order"], "schema": schema,
            "label": contract["label"], "values": contract["values"],
            "prompt_sha": sha(system + "\0" + user),
            "schema_sha": sha(json.dumps(schema, separators=(",", ":")))}


def read_text(path: str, what: str) -> str:
    try:
        with open(resolve(path), encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise ConfigError(f"{what}: {e}") from e


def load_prompt(arg: str) -> dict:
    """A name from the package, or a directory holding system.md and user.md -- a checkout,
    a folder on Drive, anywhere."""
    d = resolve(arg) if os.sep in arg else os.path.join(PROMPTS, arg)
    if not os.path.isdir(d) and os.sep not in arg:
        raise ConfigError(f"prompt {arg!r}: no such prompt in the package. It ships "
                          f"{sorted(os.listdir(PROMPTS))}; a path to a directory with "
                          f"system.md and user.md works too, as do --system and --user")
    system = read_text(os.path.join(d, "system.md"), f"prompt {arg!r}")
    user = read_text(os.path.join(d, "user.md"), f"prompt {arg!r}")
    return build_prompt(os.path.basename(d.rstrip(os.sep)), system, user, d)


def example_block(system: str) -> str:
    """The prompt's JSON example of the answer: a line that starts with `{`, up to its
    closing brace.

    This is the contract. Whatever shape the example shows is the shape the judge demands,
    so nothing about the answer -- its keys, their order, or the values a label may take --
    is written into this repository.

    A prompt may show the model several JSON objects: worked examples, a good answer and
    a bad one. Those are between the prompt and the model. The judge cannot tell which of
    them is the contract, so the prompt says which, by fencing it in ```answer.
    """
    lines, blocks, depth, buf, start = system.splitlines(), [], 0, [], 0
    for i, line in enumerate(lines):
        if depth == 0:
            if not line.lstrip().startswith("{"):
                continue
            start = i
        buf.append(line)
        depth += braces(line)
        if depth <= 0:
            blocks.append((start, "\n".join(buf)))
            depth, buf = 0, []
    if not blocks:
        raise ConfigError("the system prompt must show the answer as a JSON example, on "
                          "its own line(s) starting with '{'; found none")
    if len(blocks) == 1:
        return blocks[0][1]
    marked = [b for i, b in blocks if i and lines[i - 1].strip().lower() == ANSWER_FENCE]
    if len(marked) == 1:
        return marked[0]
    raise ConfigError(
        f"the system prompt shows {len(blocks)} JSON objects and {len(marked)} of them are "
        f"marked as the answer, so the judge cannot tell which one it must demand. Show "
        f"the model as many examples as you like, and fence the one that is the contract:\n"
        f"```answer\n{{\"verdict\": \"consistent\" | \"inconsistent\", \"why\": \"<15 words>\"}}\n```")


def braces(line: str) -> int:
    """Net `{` minus `}` outside string literals."""
    depth, quoted, escaped = 0, False, False
    for ch in line:
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            quoted = not quoted
        elif not quoted:
            depth += (ch == "{") - (ch == "}")
    return depth


BARE = re.compile(r'(?<!["\w])<[^>]*>(?!")')


def fold_alternatives(text: str) -> str:
    """`"a" | "b" | "c"` -> `{"|": ["a", "b", "c"]}`, so the example becomes valid JSON.

    A scanner rather than a regex, because the alternatives may be strings holding commas,
    braces or a `|` of their own."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] == '"':
            j = skip_string(text, i)
            out.append(text[i:j])
            i = j
        elif text[i] != ":":
            out.append(text[i])
            i += 1
        else:
            j, depth, cuts = i + 1, 0, []
            while j < n:
                c = text[j]
                if c == '"':
                    j = skip_string(text, j)
                    continue
                if c in "{[":
                    depth += 1
                elif c in "}]":
                    if depth == 0:
                        break
                    depth -= 1
                elif depth == 0 and c == ",":
                    break
                elif depth == 0 and c == "|":
                    cuts.append(j)
                j += 1
            value = text[i + 1:j]
            if cuts:
                parts = []
                for a, b in zip([i] + cuts, cuts + [j]):
                    parts.append(text[a + 1:b].strip())
                out.append(': {"|": [' + ", ".join(parts) + "]}")
            else:
                out.append(":" + fold_alternatives(value))   # an object or array nests
            i = j
    return "".join(out)


def skip_string(text: str, i: int) -> int:
    """-> the index just past the string literal starting at `text[i] == '\"'`."""
    j = i + 1
    while j < len(text) and text[j] != '"':
        j += 2 if text[j] == "\\" else 1
    return j + 1


def example_to_json(block: str):
    """The example is not valid JSON -- it carries `"a" | "b"` alternatives and `<...>`
    placeholders. Fold both into JSON, so the shape can be read with json.loads instead of
    a hand-written parser."""
    text = fold_alternatives(BARE.sub(lambda m: json.dumps(m.group(0)), block))
    try:
        return json.loads(text)
    except ValueError as e:
        raise ConfigError(f"the JSON example in the system prompt cannot be read ({e}). "
                          f"Values may be literals, <placeholders>, or alternatives "
                          f'written "a" | "b" | "c".\n{block}') from e


def schema_of(node) -> dict:
    """A JSON Schema for one node of the example. Nothing here is judge-specific."""
    if isinstance(node, dict) and set(node) == {"|"}:
        values = node["|"]
        if all(isinstance(v, str) and v.startswith("<") for v in values):
            return {"type": "string"}
        kinds = {"string" if isinstance(v, str) else
                 "boolean" if isinstance(v, bool) else "number" for v in values}
        return {"type": kinds.pop() if len(kinds) == 1 else "string", "enum": values}
    if isinstance(node, dict):
        return {"type": "object",
                "properties": {k: schema_of(v) for k, v in node.items()},
                "required": list(node), "additionalProperties": False}
    if isinstance(node, list):
        if not node:
            raise ConfigError("an empty array in the JSON example says nothing about the "
                              "answer; show one element")
        return {"type": "array", "items": schema_of(node[0])}
    if isinstance(node, bool):
        return {"type": "boolean"}
    if isinstance(node, (int, float)):
        return {"type": "number"}
    return {"type": "string"}


def read_contract(system: str) -> dict:
    """-> {order, schema, label, values, canary_key}: what the prompt asks the model to
    answer.

    `label` is the first key whose example lists alternatives -- the categorical answer
    the summary counts and the agreement check compares. `verdict` in the prompts that
    ship here, `score` or `category` in yours; the name is the prompt's business, not
    this repo's.

    A prompt need not have one. An answer that is a free number, or prose and nothing
    else, is recorded whole like any other; there is simply nothing to count, so the
    summary reports how many rows were answered and leaves the rates out.
    """
    example = example_to_json(example_block(system))
    if not isinstance(example, dict):
        raise ConfigError("the JSON example must be an object, not a "
                          f"{type(example).__name__}")
    schema = schema_of(example)
    order = tuple(example)
    labels = [k for k in order if "enum" in schema["properties"][k]]
    first = FIRST.findall(system)
    if first and first[-1] != order[0]:
        raise ConfigError(f'the prompt says write "{first[-1]}" FIRST but its JSON example '
                          f'starts with "{order[0]}"')
    return {"order": order, "schema": schema,
            "label": labels[0] if labels else None,
            "values": schema["properties"][labels[0]]["enum"] if labels else [],
            # What the canary pins to one value. The label where there is one; otherwise
            # the first key, which is enough to catch a server ignoring response_format.
            "canary_key": labels[0] if labels else order[0]}


def canary_schema(contract: dict) -> dict:
    """The contract's schema with one key pinned to one value, so a server that ignores
    `response_format` cannot produce it by accident."""
    schema = json.loads(json.dumps(contract["schema"]))
    schema["properties"][contract["canary_key"]] = {"type": "string", "enum": [CANARY]}
    return schema


def check_answer(obj, contract: dict) -> dict:
    """Shared by both parsers: the top-level keys the prompt asked for, and a label the
    prompt allows. Depth below that is the schema's job under guided decoding, and not
    worth guessing at without one."""
    label = contract["label"]
    if set(obj) != set(contract["order"]):
        return {"parse_error": f"keys {sorted(obj)}"}
    if label is None:
        # Nothing in this answer is categorical, so the keys are the whole check and the
        # answer is recorded as it came.
        return {"verdict": None, "answer": obj,
                "short_reason": obj.get("short_reason") if isinstance(
                    obj.get("short_reason"), str) else None,
                "key_order_ok": tuple(obj) == contract["order"]}
    value = obj[label]
    if isinstance(value, str):
        value = value.strip().lower() if isinstance(contract["values"][0], str) else value
    if value not in contract["values"]:
        return {"parse_error": f"{label} {obj[label]!r}"}
    return {"verdict": value, "answer": obj,
            "short_reason": obj.get("short_reason") if isinstance(
                obj.get("short_reason"), str) else None,
            "key_order_ok": tuple(obj) == contract["order"]}


def parse_strict(content: str | None, contract: dict) -> dict:
    """json.loads, exactly the keys the prompt asked for. No fallbacks: with a schema, a
    reply that needs one means the server is not enforcing it."""
    try:
        obj = json.loads(content or "")
    except ValueError as e:
        return {"parse_error": f"not JSON: {str(e)[:80]}"}
    if not isinstance(obj, dict):
        return {"parse_error": f"JSON {type(obj).__name__}, not an object"}
    return check_answer(obj, contract)


def parse_tolerant(content: str | None, contract: dict) -> dict:
    """Without a grammar: the LAST JSON object carrying the label key, and the text before it.

    Taking the last such object, rather than everything between the first `{` and the
    last `}`, tolerates plain-text reasoning and code fences around the answer. Whatever
    the model wrote before it is kept as `reasoning`.

    MedGemma's `<unused94>thought` span is dropped first, so draft JSON inside it cannot
    beat the answer that follows. A span that never closes is the exception, and
    `parse_note` records it.

    The keys and the label value are checked as strictly as on the guided path: a reply
    that misses either is a parse error, never a guessed answer.
    """
    raw = content or ""
    body, note = THOUGHT.sub("", raw), None
    if raw.rfind("<unused94>") > raw.rfind("<unused95>"):
        # An unclosed thought span runs to the end of the reply, so THOUGHT would strip
        # the answer along with it. Scan the whole reply instead, markers removed.
        body = raw.replace("<unused94>", "").replace("<unused95>", "")
        note = "json inside unclosed thought"
    label = contract["label"]
    decoder, found, at = json.JSONDecoder(), None, 0
    for m in re.finditer(r"\{", body):
        try:
            obj, _ = decoder.raw_decode(body, m.start())
        except ValueError:
            continue
        # With a label, the answer is the object carrying it -- that is the one key the
        # reasoning above cannot accidentally produce. Without one, all of the keys.
        if isinstance(obj, dict) and (label in obj if label else
                                      all(k in obj for k in contract["order"])):
            found, at = obj, m.start()
    if found is None:
        return {"parse_error": f"no JSON object with a "
                               f"{label or ' and a '.join(contract['order'])}: "
                               f"{body.strip()[:80]!r}"}
    parsed = check_answer(found, contract)
    parsed.update(reasoning=body[:at].strip() or None, parse_note=note)
    return parsed


def build_payload(model: str, prompt: dict, user_text: str, decoding: dict,
                  extra: dict | None = None, schema: dict | None = None,
                  name: str = "verdict", guided: bool = True,
                  drop: list[str] | None = None, stream: bool = False) -> dict:
    """The request body. `extra` adds or overrides fields, `drop` removes them, and a
    field set to null is removed too.

    Not every OpenAI-compatible API takes the same fields. Anthropic has no `seed`, the
    reasoning models reject `temperature`, and newer OpenAI models want
    `max_completion_tokens` instead of `max_tokens`. Renaming is dropping plus adding:

        --drop max_tokens --extra '{"max_completion_tokens": 4096}'
    """
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
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}   # or no token counts at all
    payload.update(extra or {})
    for k in list(drop or []) + [k for k, v in payload.items() if v is None]:
        payload.pop(k, None)
    return payload


# --------------------------------------------------------------------------------------
# pool


class Item:
    __slots__ = ("id", "fields", "group", "stratum", "weight",
                 "key", "attempts", "parse_retries", "auth_retried")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        self.attempts, self.parse_retries, self.auth_retried = 0, 0, False

    def clone(self, key: str) -> Item:
        return Item(**{k: getattr(self, k) for k in self.__slots__ if k != "key"}, key=key)


def load_items(path: str, seed: int, columns: tuple[str, ...] = ()) -> tuple[list[Item], str, str]:
    """-> (items in send order, sha of the items file, order.csv text).

    The file is read whole and checked whole before a single request goes out: a pool that
    is malformed on line 40,000 should cost nothing, not four hours. `columns` are the
    `{column}` placeholders the prompt names; every row must carry all of them, because a
    row that cannot be rendered is discovered at render time, mid-run, with the model
    already warm.
    """
    items: list[Item] = []
    seen: set[str] = set()
    try:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ConfigError(f"{path}:{lineno}: not JSON ({e})") from e
                if not isinstance(obj, dict):
                    raise ConfigError(f"{path}:{lineno}: expected an object, got {type(obj).__name__}")
                item_id = obj.get("id")
                if not isinstance(item_id, str) or not item_id:
                    raise ConfigError(f'{path}:{lineno}: "id" must be a non-empty string')
                if item_id in seen:
                    raise ConfigError(f"{path}:{lineno}: duplicate id {item_id!r}")
                seen.add(item_id)
                fields = obj.get("fields")
                if not isinstance(fields, dict):
                    raise ConfigError(f'{path}:{lineno}: "fields" must be an object')
                absent = [c for c in columns if c not in fields]
                if absent:
                    raise ConfigError(f"{path}:{lineno}: item {item_id!r} lacks the fields "
                                      f"the prompt names: {absent}")
                weight = obj.get("weight", 1.0)
                if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                    raise ConfigError(f'{path}:{lineno}: "weight" must be a number')
                items.append(Item(id=item_id, fields={k: v for k, v in fields.items()},
                                  group=str(obj.get("group", "")),
                                  stratum=str(obj.get("stratum", "")),
                                  weight=float(weight)))
    except FileNotFoundError as e:
        raise ConfigError(f"missing items file {path}") from e
    if not items:
        raise ConfigError(f"{path}: no items")

    items.sort(key=lambda it: it.id)
    random.Random(f"{seed}|order").shuffle(items)

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(ORDER_FIELDS)
    for i, it in enumerate(items):
        w.writerow([i, it.id, it.group, it.stratum, it.weight])
    return items, file_sha(path), buf.getvalue()


def row_key(item_id: str, model: str, prompt: dict,
            endpoint: str = "", guided: bool = True) -> str:
    """What resume is keyed on. Changing any part of it means a row is sent again.

    The items file's sha is deliberately not in here. Pools grow: add 200 rows to the
    file, re-run, and only those 200 should be sent. Keying on the file would re-judge
    every row already on disk. The id is unique within the file, and the prompt, model
    and schema are pinned separately, so the file's identity adds nothing.
    """
    return sha("|".join([item_id, model, prompt["prompt_sha"],
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
                key = rec["key"]
            except (ValueError, TypeError, KeyError):
                # Torn by a hard kill, or edited by hand. Skipping it costs one row;
                # raising here would cost the whole run before it sent anything.
                bad += 1
                continue
            latest[key] = rec
            n += 1
    return latest, n, bad


# What may not change between sessions writing into one results directory. The items
# file is not on the list: a pool that grew is the ordinary case, and every row carries
# its own id. What must not change is how a row was judged.
PINNED = ("prompt", "prompt_sha", "schema_sha", "key_order", "guided",
          "order_seed", "mode")
PINNED_DEFAULTS: dict = {}
# `decoding` is checked field by field instead, and max_tokens is left out: raising the
# budget is how a run recovers its truncated rows, and a reply cut off at max_tokens is
# recorded as an error rather than a verdict, so no answer already on disk was shaped by
# the old budget. Sampling -- temperature, seed -- is pinned like everything else.
UNPINNED_DECODING = ("max_tokens",)
# Re-stated every session instead, so run.json describes the pool that is there now and
# the sessions list keeps the history.
RESTATED = ("items", "items_sha", "items_rows", "order_sha", "decoding")


def check_run_json(out: str, meta: dict, session: dict) -> None:
    path = os.path.join(out, "run.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        diff = [k for k in PINNED if old.get(k, PINNED_DEFAULTS.get(k)) != meta.get(k)]
        was, now = old.get("decoding") or {}, meta.get("decoding") or {}
        diff += [f"decoding.{k}" for k in sorted(set(was) | set(now))
                 if k not in UNPINNED_DECODING and was.get(k) != now.get(k)]
        if diff:
            raise ConfigError(
                f"{rel(out)} was started with different {diff}. Use a new --run-tag; "
                f"never pool results across configurations.")
        old.update({k: meta.get(k) for k in RESTATED})
        old.setdefault("sessions", []).append(session)
        write_json(path, old)
    else:
        write_json(path, {**meta, "created": now_iso(), "sessions": [session]})


def summarise(results_path: str, meta: dict) -> dict:
    latest, lines, bad = read_results(results_path)
    recs = list(latest.values())
    label = meta.get("label")
    values = meta.get("values") or sorted({r["verdict"] for r in recs if r.get("verdict")})
    out: dict = {"prompt": meta["prompt"], "items": meta.get("items"), "mode": meta["mode"],
                 "label": label, "values": values,
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
    for g in sorted({r.get("group") or "all" for r in recs}):
        rs = [r for r in recs if (r.get("group") or "all") == g]
        # An answered row, whether or not the prompt asked for anything countable.
        judged = [r for r in rs if r.get("answer") is not None]
        wsum = sum(r.get("weight") or 1.0 for r in judged) or 1.0

        def share(wanted, weighted):
            hit = [r for r in judged if r["verdict"] in wanted]
            if weighted:
                return round(sum(r.get("weight") or 1.0 for r in hit) / wsum, 4)
            return round(len(hit) / max(1, len(judged)), 4)

        out["by_group"][g] = {
            "rows": len(rs), "judged": len(judged),
            "verdicts": dict(collections.Counter(r["verdict"] for r in judged
                                                 if r.get("verdict") is not None)),
            "not_judged": len(rs) - len(judged),
            # One share per value the prompt allows, whatever they are called.
            "rates": {str(v): share({v}, False) for v in values},
            "rates_weighted": {str(v): share({v}, True) for v in values},
            "by_stratum": {s: dict(collections.Counter(r.get("verdict") or "not_judged"
                                                       for r in rs if r.get("stratum", "") == s))
                           for s in sorted({r.get("stratum", "") for r in rs})},
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
        by_row: dict[str, dict[str, str]] = collections.defaultdict(dict)
        for r in recs:
            if r.get("verdict"):
                by_row[r["id"]][r["endpoint"]] = r["verdict"]
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
        # The run's own --extra / --drop apply to every endpoint, on top of its entry's.
        self.extra = {**(cfg.get("extra") or {}), **(run.args.extra or {})}
        self.drop = list(cfg.get("drop") or []) + list(run.args.drop or [])
        self.metrics = bool(cfg.get("metrics", False))
        # Where this API answers. Only a vLLM-shaped server has everything under
        # <base>/v1: Anthropic answers at /messages, and an API with no model listing at
        # all sets models_path = "" -- as --skip-model-check does -- so that preflight
        # proves the endpoint with the canary alone.
        self.chat_path = cfg.get("chat_path", "/chat/completions")
        self.stream = bool(cfg.get("stream", run.args.stream))
        self.models_path = ("" if run.args.skip_model_check
                            else cfg.get("models_path", "/models"))
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
        """Where the URL, the key and the headers come from, in order: the endpoint entry
        itself (`--base-url`, `--api-key`), then the env file, then the process
        environment.

        The env file does not have to exist. A Colab cell that ran `serve_vllm.py` has
        the URL and the key in `os.environ` already, and a vendor API usually has its key
        there too."""
        env_file = resolve(self.cfg.get("env_file", ".env"))
        env = read_env(env_file) if os.path.isfile(env_file) else {}
        where = f"the endpoints file, {rel(env_file)} or the environment"
        url = self.cfg.get("url") or env.get(self.cfg["url_key"]) or \
            os.environ.get(self.cfg["url_key"])
        if not url:
            raise ConfigError(f"endpoint {self.name}: no URL — {self.cfg['url_key']} is "
                              f"in none of {where}")
        url = (url if "://" in url else "https://" + url).rstrip("/")
        parts = urllib.parse.urlsplit(url)
        # A URL that already carries a path is used exactly as given: Azure's
        # /openai/deployments/<name>, Gemini's /v1beta/openai, a gateway's prefix. Only
        # a bare host gets /v1 appended.
        api = url if parts.path.strip("/") else join(url, "/v1")
        root = f"{parts.scheme}://{parts.netloc}"        # vLLM serves /metrics at the root
        headers = dict(HEADERS)
        if self.cfg.get("auth_key"):
            cred = (self.cfg.get("api_key") or env.get(self.cfg["auth_key"])
                    or os.environ.get(self.cfg["auth_key"]))
            if not cred:
                raise ConfigError(f"endpoint {self.name}: no key — {self.cfg['auth_key']} "
                                  f"is in none of {where}")
            # Anthropic sends the key raw in x-api-key and Azure raw in api-key, so a
            # header that is not Authorization takes the key as it is unless the entry
            # says otherwise.
            name = self.cfg.get("auth_header", "Authorization")
            scheme = self.cfg.get("auth_scheme") or ("bearer" if name == "Authorization"
                                                     else "raw")
            if scheme not in ("bearer", "basic", "raw"):
                raise ConfigError(f"endpoint {self.name}: auth_scheme {scheme!r} is not "
                                  f"one of bearer, basic, raw")
            headers[name] = {
                "bearer": f"Bearer {cred}",
                "basic": "Basic " + base64.b64encode(cred.encode()).decode(),
                "raw": cred}[scheme]
        # The entry's own headers, then the run's --header, on top of both.
        headers.update({str(k): str(v) for k, v in (self.cfg.get("headers") or {}).items()})
        headers.update(self.run.args.header or {})
        self.root, self.api = root, api
        self.headers = {k: v for k, v in headers.items() if v != ""}   # "" removes one

    def event(self, text: str, loud: bool = True) -> None:
        self.events.append(f"{now_iso()} {text}")
        if loud:
            log(f"{self.name}: {text}")

    # -- transport -------------------------------------------------------------------
    @staticmethod
    def transport_failure(e: Exception) -> tuple[str, str]:
        if isinstance(e, httpx.ConnectTimeout):
            return "down", "connect timeout"
        if isinstance(e, httpx.TimeoutException):
            return "timeout", type(e).__name__
        return "down", f"{type(e).__name__}: {str(e)[:160]}"

    @staticmethod
    def http_failure(status: int, headers, text: str) -> tuple[str, str]:
        text = text[:400]
        m = NGROK_CODE.search(text)
        code = headers.get("ngrok-error-code") or (m.group(0) if m else None)
        return (classify(status, code),
                f"HTTP {status}{' ' + code if code else ''}: {text[:200]}")

    def before_request(self) -> tuple[str, str] | None:
        if self.api is None:
            return "down", f"no URL: {self.cfg['url_key']} is not in the env file"
        stop = self.run.spend_one()
        if stop:
            return stop
        self.stats["requests"] += 1
        return None

    async def call(self, method: str, path: str, body: bytes | None, timeout: float,
                   stream: bool = False) -> tuple[str, object]:
        """-> ("ok", reply) or (failure class, detail). Never raises for the network.

        `reply` is the httpx response, or -- streamed -- the chat completion assembled
        from the events, in the shape the same call returns unstreamed."""
        stop = self.before_request()
        if stop:
            return stop
        url = join(self.root if path == "/metrics" else self.api, path)
        if stream:
            return await self.call_stream(url, body, timeout)
        try:
            r = await self.client.request(
                method, url, content=body, headers=self.headers,
                timeout=httpx.Timeout(timeout, connect=min(15.0, timeout)))
        except httpx.TransportError as e:
            return self.transport_failure(e)
        if r.status_code == 200:
            return "ok", r
        return self.http_failure(r.status_code, r.headers, r.text)

    async def call_stream(self, url: str, body: bytes | None, timeout: float
                          ) -> tuple[str, object]:
        """The same call, read as server-sent events and assembled into one reply.

        Cloudflare cuts a request whose origin has sent nothing for about 100 s (524), and
        a prompt that reasons before it answers can spend longer than that on a reply the
        server is still buffering: the tunnel kills a row the model is still working on.
        Streamed, the timeout measures silence rather than the whole reply, so --timeout
        can go past the tunnel's limit and a long answer still arrives.
        """
        text, finish, usage = [], None, {}
        try:
            async with self.client.stream(
                    "POST", url, content=body, headers=self.headers,
                    timeout=httpx.Timeout(timeout, connect=min(15.0, timeout))) as r:
                if r.status_code != 200:
                    detail = (await r.aread()).decode("utf-8", "replace")
                    return self.http_failure(r.status_code, r.headers, detail)
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue                  # a comment or a keep-alive, not an event
                    for choice in chunk.get("choices") or []:
                        text.append((choice.get("delta") or {}).get("content") or "")
                        finish = choice.get("finish_reason") or finish
                    usage = chunk.get("usage") or usage
        except httpx.TransportError as e:
            return self.transport_failure(e)
        return "ok", {"choices": [{"index": 0, "finish_reason": finish,
                                   "message": {"role": "assistant",
                                               "content": "".join(text)}}],
                      "usage": usage}

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
        """The model listing, where there is one, then a canary. Guided, the canary proves
        the schema is enforced; unguided there is nothing to enforce, so it only has to
        parse. An API that lists no models -- or lists deployment names rather than the
        name you send -- skips the listing and is proven by the canary alone."""
        if self.api is None:
            try:
                self.load_env()
            except ConfigError as e:
                return self.set_down(f"waiting for its URL ({e})")
        if self.models_path:
            kind, r = await self.call("GET", self.models_path, None, 30)
            if kind == "auth":
                self.reload_env()
                kind, r = await self.call("GET", self.models_path, None, 30)
            if kind != "ok":
                return self.set_down(f"{self.models_path}: {r}")
            try:
                self.served = [m["id"] for m in r.json()["data"]]
            except (ValueError, KeyError, TypeError):
                return self.set_down(f"{self.models_path}: unexpected payload")
            if self.model not in self.served:
                return self.refuse(f"model {self.model!r} is not served; server has "
                                   f"{self.served}")

        run = self.run
        contract = run.prompt["contract"]
        order, pinned = contract["order"], contract["canary_key"]
        payload = build_payload(
            self.model, run.prompt, run.render(run.canary_item), run.decoding, self.extra,
            schema=canary_schema(contract) if run.guided else None,
            name="canary", guided=run.guided, drop=self.drop, stream=self.stream)
        kind, r = await self.call("POST", self.chat_path, json.dumps(payload).encode(),
                                  run.args.timeout, stream=self.stream)
        if kind == "auth":
            self.reload_env()
            kind, r = await self.call("POST", self.chat_path,
                                      json.dumps(payload).encode(), run.args.timeout,
                                      stream=self.stream)
        if kind == "bad_request":
            return self.refuse(f"canary request rejected: {r}")
        if kind != "ok":
            return self.set_down(f"canary: {r}")
        try:
            data = r if isinstance(r, dict) else r.json()
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
            if not isinstance(obj, dict) or obj.get(pinned) != CANARY:
                return self.refuse(f"server ignores response_format: canary {pinned} "
                                   f"{obj.get(pinned) if isinstance(obj, dict) else obj!r}")
            if tuple(obj) != tuple(order):
                return self.refuse(f"server does not enforce key order: got {list(obj)}, "
                                   f"schema {list(order)}")
        else:
            # Nothing to enforce, and one row is too little to judge a model on, so a canary
            # that does not parse is a warning; the parse-error breaker in handle() decides.
            parsed = parse_tolerant(content, contract)
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
            last = time.monotonic()
            if await self.preflight() != "down":
                return
            if not changed:
                delay = min(args.probe_max, delay * 2)


# --------------------------------------------------------------------------------------
# run


class Run:
    def __init__(self, args, prompt: dict, decoding: dict, meta: dict, out: str,
                 items_by_ep: dict[str, list[Item]], shared: bool,
                 endpoint_cfgs: list[dict], done: set[str], canary_item: Item):
        self.args, self.prompt, self.decoding, self.meta, self.out = (
            args, prompt, decoding, meta, out)
        self.guided = args.guided == "on"
        self.done = done
        self.canary_item = canary_item
        self.shared = shared
        self.session = now_iso()
        self.stopping = False
        self.stop_reason = ""
        self.requests = 0
        self.max_requests = int(args.max_requests or 0)
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
        return render_user(self.prompt["user"], it.fields)

    def queues(self) -> list[collections.deque]:
        seen, out = set(), []
        for ep in self.endpoints:
            if id(ep.queue) not in seen and (self.shared or ep.state != "refused"):
                seen.add(id(ep.queue))
                out.append(ep.queue)
        return out

    def spend_one(self) -> tuple[str, str] | None:
        """The run's request ceiling. Every request passes through here -- rows, preflight
        and probes alike -- because a ceiling that counted only rows would not be one.
        Reaching it stops the run the way Ctrl-C does: the requests in flight finish, the
        summary is written, and the rows never sent are reported as missing."""
        if self.max_requests and self.requests >= self.max_requests:
            self.stop(f"--max-requests {self.max_requests} reached")
            return "limit", f"the run's ceiling of {self.max_requests} requests"
        self.requests += 1
        return None

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
            "id": it.id, "group": it.group, "stratum": it.stratum, "weight": it.weight,
            "endpoint": ep.name, "model": ep.model, "prompt": self.prompt["name"],
            "prompt_sha": self.prompt["prompt_sha"], "schema_sha": self.prompt["schema_sha"],
            "guided": self.guided,
            "decoding": self.decoding, "extra": ep.extra or None,
            "items_sha": self.meta["items_sha"],
            "label": self.prompt["label"], "answer": parsed.get("answer"),
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
        it.fields = None                       # the pool can be large; free it once recorded

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
        t0 = time.monotonic()
        try:
            body = json.dumps(build_payload(ep.model, self.prompt, self.render(it),
                                            self.decoding, ep.extra, guided=self.guided,
                                            drop=ep.drop, stream=ep.stream)).encode()
            kind, r = await ep.call("POST", ep.chat_path, body, self.args.timeout,
                                    stream=ep.stream)
        except Exception as e:
            # Rendering and serialising happen here. A bug in either must not kill the
            # task before the in-flight slot is returned, or remaining() never reaches
            # zero and the run hangs with rows left. Record the row and move on.
            self.record(ep, it, None, {"latency_s": round(time.monotonic() - t0, 3)},
                        error=("internal", f"{type(e).__name__}: {str(e)[:200]}"))
            ep.errors["internal"] += 1
            return
        finally:
            ep.inflight -= 1
        latency = round(time.monotonic() - t0, 3)

        if kind == "ok":
            try:
                data = r if isinstance(r, dict) else r.json()
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
                parsed = parse(content, self.prompt["contract"])
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
        if kind == "limit":
            q.appendleft(it)              # the run is already stopping; keep the row unsent
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
            live = [ep for ep in self.endpoints if ep.state != "refused"
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
        write_json(os.path.join(self.out, "endpoints.json"), {
            "session": self.session, "updated": now_iso(),
            "requests": self.requests, "max_requests": self.max_requests or None,
            "endpoints": [{
                "name": ep.name, "env_file": ep.cfg.get("env_file", ".env"),
                "url_key": ep.cfg["url_key"], "auth_key": ep.cfg.get("auth_key") or None,
                "model": ep.model, "served": ep.served, "state": ep.state,
                "api": ep.api, "chat_path": ep.chat_path, "stream": ep.stream,
                "requests_this_session": ep.stats["requests"], "rows_this_session": ep.stats["rows"],
                "window": ep.tuner.limit, "window_bounds": [ep.tuner.lo, ep.tuner.hi],
                "errors": dict(ep.errors), "extra": ep.extra or None,
                "drop": ep.drop or None, "canary": ep.canary,
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
            self.write_endpoints()
        s = self.write_summary()
        log(f"wrote {self.written:,} rows this session; {s['missing']:,} of {s['planned']:,} "
            f"planned rows still missing; {s['parse_errors']} parse errors, errors "
            f"{s['errors']}; duplicates suppressed {self.duplicates}")
        return EXIT_OK if s["missing"] == 0 else EXIT_INCOMPLETE


# --------------------------------------------------------------------------------------
# entry point


def run_async(coro):
    """asyncio.run, except in a notebook cell, where there is already a loop running.

    Jupyter and Colab run a cell inside their own event loop, where `asyncio.run` raises
    "cannot be called from a running event loop". A worker thread gets a loop of its own
    and the cell blocks on the join, which is what a cell should do anyway. Signal
    handlers are skipped off the main thread, so Ctrl-C there is the kernel's interrupt
    rather than a drain.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict = {}

    def work():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as e:                      # noqa: BLE001 - re-raised below
            box["error"] = e

    t = threading.Thread(target=work, name="llmjudge")
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def cli_endpoint(args) -> list[dict]:
    """One endpoint straight off the command line: no endpoints file, no .env.

    This is how a colleague with a vendor API key, or a Colab cell that just started its
    own server, runs the judge -- two flags rather than two files to write first."""
    if not args.model:
        raise ConfigError("--base-url needs --model: the name the server answers "
                          "/v1/models with")
    cfg = {"name": "cli", "url_key": "LLMJUDGE_BASE_URL", "model": args.model,
           "env_file": os.devnull}
    if args.base_url:
        cfg["url"] = args.base_url
    if args.chat_path:
        cfg["chat_path"] = args.chat_path
    if args.auth_header:
        cfg["auth_header"] = args.auth_header
    key = args.api_key or os.environ.get("LLMJUDGE_API_KEY")
    if key:                                # no key at all is fine: a local server needs none
        cfg["auth_key"], cfg["api_key"] = "LLMJUDGE_API_KEY", key
    return [cfg]


def load_endpoints(path: str, only: list[str]) -> list[dict]:
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"no endpoints file {rel(path)}. For a single server, "
                          f"--base-url https://host/v1 --model NAME needs no file at "
                          f"all; for several, copy configs/endpoints.example.toml") from e
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
    return eps


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompt",
                    help="required: a prompt that ships with the package (c1, c2, c3, ...), "
                         "or a path to any directory holding system.md and user.md. There "
                         "is no default -- the prompt is what decides what is judged. "
                         "--system and --user give one instead")
    ap.add_argument("--system", help="path to your own system prompt; needs --user too, and "
                                     "then --prompt is ignored")
    ap.add_argument("--user", help="path to your own user template, the one with the "
                                   "{column} placeholders")
    ap.add_argument("--prompt-name", default="custom",
                    help="what to call a --system/--user prompt in run.json and summary.json")
    ap.add_argument("--items", required=True,
                    help="JSONL, one object per row: id, fields, and optionally group, "
                         "stratum, weight")
    ap.add_argument("--seed", type=int, default=42, help="send-order and decoding seed")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--endpoints", default=ENDPOINTS_DEFAULT,
                    help="TOML file of endpoints, for a run against more than one server. "
                         "--base-url instead for a single one, and then no file is needed")
    ap.add_argument("--endpoint", action="append", default=[],
                    help="use only this endpoint (repeatable)")
    ap.add_argument("--base-url", help="one endpoint, with no endpoints file and no .env: "
                                       "the OpenAI-compatible base URL, e.g. "
                                       "https://host/v1. Also read from LLMJUDGE_BASE_URL")
    ap.add_argument("--model", help="the model to ask for; required with --base-url")
    ap.add_argument("--api-key", help="bearer token for --base-url. Prefer the environment: "
                                      "LLMJUDGE_API_KEY is read when this is not given, and "
                                      "a key on the command line lands in your shell history")
    ap.add_argument("--extra", help="JSON object merged into every request body, e.g. "
                                    "'{\"max_completion_tokens\": 4096}'. null removes a field")
    ap.add_argument("--drop", action="append", default=[], metavar="FIELD",
                    help="remove a field from every request body (repeatable). --drop seed "
                         "--drop temperature for an API that rejects them")
    ap.add_argument("--header", action="append", default=[], metavar="NAME:VALUE",
                    help="a header on every request, over every endpoint (repeatable): "
                         "--header 'anthropic-version: 2023-06-01'. An empty value removes "
                         "a header the judge would otherwise send")
    ap.add_argument("--auth-header",
                    help="the header the key goes in when it is not Authorization: Bearer. "
                         "x-api-key for Anthropic, api-key for Azure; the key is then sent "
                         "raw, without a scheme")
    ap.add_argument("--chat-path", metavar="PATH",
                    help="the path appended to the base URL, default /chat/completions. "
                         "/messages for Anthropic")
    ap.add_argument("--stream", action="store_true",
                    help="read replies as server-sent events instead of waiting for the "
                         "whole body. Behind a tunnel that cuts a silent request (a "
                         "Cloudflare 524 at ~100 s) this is what lets a slow reply finish, "
                         "and --timeout then measures silence rather than the whole reply. "
                         "Per endpoint: stream = true")
    ap.add_argument("--max-requests", type=int, default=0, metavar="N",
                    help="stop the run after N requests: the ceiling that keeps a typo "
                         "from judging 55,000 rows against a paid API. Rows, preflight "
                         "and probes all count. The run stops the way Ctrl-C stops it, so "
                         "the rows already answered are on disk and a re-run resumes")
    ap.add_argument("--skip-model-check", action="store_true",
                    help="do not ask /models whether the model is served, for an API that "
                         "has none or that lists deployment names. The canary still has to "
                         "come back sound, so the endpoint is still proven before any row")
    ap.add_argument("--mode", choices=["shard", "compare"], default="shard")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows of the send order")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-send rows whose latest record is an error or parse error")
    ap.add_argument("--guided", choices=["on", "off"], default="on",
                    help="on: response_format carries the schema read from the prompt and the reply must be "
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
                    help="write the send order, prompt and request examples; call nothing")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "make-items":
        from . import items
        return items.main(argv[1:])
    args = parse_args(argv)
    try:
        return run_main(args)
    except ConfigError as e:
        print(f"llmjudge: refused: {e}", file=sys.stderr, flush=True)
        return EXIT_CONFIG


def judge(items: str, out: str, run_tag: str, system: str | None = None,
          user: str | None = None, **options) -> int:
    """The notebook entry point. Same options as the command line, with underscores.

        from llmjudge import judge
        judge(items="/content/drive/MyDrive/judge/items.jsonl",
              out="/content/drive/MyDrive/judge/results/pilot",
              run_tag="pilot-01", system=SYSTEM, user=USER, guided="off", limit=100,
              base_url="http://127.0.0.1:8000/v1", model="medgemma-27b-it")

    `base_url` and `model` are one endpoint with no files to write first; the key comes
    from `api_key=` or the environment. A dict option is sent as JSON and a list option
    repeats its flag, so `extra={"max_completion_tokens": 4096}` and `drop=["seed"]`
    both work from a cell.

    `system` and `user` are the prompt itself -- the text of a cell, or a path to a file
    on Drive; a value that names an existing file is read, anything else is the prompt.
    Give neither and `prompt="c3"` picks one of the prompts that ship with the package,
    or `prompt="/content/drive/MyDrive/my-prompt"` a directory of your own.

    Prompt text is written to `<out>/prompt/`, so the results directory always carries
    the exact prompt that produced it, and a re-run from the same cell resumes.
    """
    argv = ["--items", str(items), "--out", str(out), "--run-tag", str(run_tag)]
    if (system is None) != (user is None):
        print("llmjudge: refused: system= and user= go together", file=sys.stderr)
        return EXIT_CONFIG
    if system is not None:
        d = os.path.join(resolve(out), "prompt")
        os.makedirs(d, exist_ok=True)
        for name, text in (("system.md", system), ("user.md", user)):
            path = os.path.join(d, name)
            text = read_text(text, name) if _looks_like_a_path(text) else text
            with open(path, "w", encoding="utf-8") as f:
                f.write(text.strip() + "\n")
            argv += [f"--{name.removesuffix('.md')}", path]
    for k, v in options.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv += [flag]
        elif isinstance(v, dict):
            argv += [flag, json.dumps(v)]            # extra={"max_completion_tokens": 4096}
        elif isinstance(v, (list, tuple)):
            for one in v:                            # drop=["seed", "temperature"]
                argv += [flag, str(one)]
        else:
            argv += [flag, str(v)]
    return main(argv)


def _looks_like_a_path(text: str) -> bool:
    """A prompt is many lines; a path is one short line that exists on disk."""
    return "\n" not in text.strip() and len(text) < 4096 and os.path.isfile(resolve(text))


def run_main(args: argparse.Namespace) -> int:
    if bool(args.system) != bool(args.user):
        raise ConfigError("--system and --user go together: one is the system prompt, the "
                          "other the user template with the {column} placeholders")
    if args.system:
        prompt = build_prompt(args.prompt_name,
                              read_text(args.system, "--system"),
                              read_text(args.user, "--user"))
    elif args.prompt:
        prompt = load_prompt(args.prompt)
    else:
        raise ConfigError(
            f"--prompt is required: one that ships with the package "
            f"({', '.join(sorted(os.listdir(PROMPTS)))}), a path to a directory holding "
            f"system.md and user.md, or --system and --user with your own. There is no "
            f"default, because the prompt is what decides what is being judged.")
    args.header = parse_headers(args.header)
    guided = args.guided == "on"
    if args.max_tokens is None:
        args.max_tokens = 96 if guided else 1024
    if isinstance(args.extra, str):
        try:
            args.extra = json.loads(args.extra)
        except ValueError as e:
            raise ConfigError(f"--extra is not JSON ({e}): {args.extra}") from e
    if args.extra is not None and not isinstance(args.extra, dict):
        raise ConfigError("--extra must be a JSON object of request fields")
    items_path = resolve(args.items)
    out = resolve(args.out)
    decoding = {"temperature": args.temperature, "seed": args.seed,
                "max_tokens": args.max_tokens}

    columns = tuple(dict.fromkeys(PLACEHOLDER.findall(prompt["user"])))
    items, items_sha, order_csv = load_items(items_path, args.seed, columns)
    meta = {"prompt": prompt["name"],
            "prompt_dir": rel(prompt["dir"]) if prompt["dir"] else None,
            "prompt_sha": prompt["prompt_sha"], "key_order": list(prompt["order"]),
            "guided": guided,
            # Unguided nothing is sent, but the sha still pins the key order the prompt
            # asks for, so it stays part of the resume key.
            "schema": prompt["schema"] if guided else None,
            "schema_sha": prompt["schema_sha"],
            "label": prompt["label"], "values": prompt["values"],
            "decoding": decoding, "order_seed": args.seed, "order_sha": sha(order_csv),
            "items": rel(items_path), "items_sha": items_sha, "items_rows": len(items),
            "mode": args.mode}
    session = {"started": now_iso(), "argv": sys.argv[1:] if __name__ == "__main__" else None,
               "script_sha": file_sha(os.path.abspath(__file__)), "dry_run": args.dry_run,
               "pid": os.getpid(),
               # The pool may differ from session to session; which one this session ran.
               "items": rel(items_path), "items_sha": items_sha, "items_rows": len(items),
               # max_tokens may be raised between sessions to recover truncated rows.
               "decoding": decoding}
    os.makedirs(out, exist_ok=True)
    order_path = os.path.join(out, "order.csv")
    with open(order_path + ".tmp", "w", encoding="utf-8", newline="") as f:
        f.write(order_csv)
    os.replace(order_path + ".tmp", order_path)

    counts = collections.Counter((it.group, it.stratum) for it in items)
    log(f"items {rel(items_path)}: {len(items):,} rows -> {rel(order_path)}")
    for (g, st), c in sorted(counts.items()):
        log(f"  {g or '-':<9} {st or '-':<28} {c:>6}")
    counted = (f"{prompt['label']} one of {prompt['values']}" if prompt["label"] else
               "no key offers a choice, so nothing is counted and the canary pins "
               f"{prompt['contract']['canary_key']!r}")
    log(f"prompt {prompt['name']}: answers {list(prompt['order'])}, {counted}, guided "
        f"decoding {'on' if guided else 'off'}, max_tokens {args.max_tokens}, prompt sha "
        f"{prompt['prompt_sha'][:16]}, schema sha {prompt['schema_sha'][:16]}")

    user_text = render_user(prompt["user"], items[0].fields)
    with open(os.path.join(out, "prompt_example.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== SYSTEM ===\n{prompt['system']}\n\n=== USER ===\n{user_text}\n")
    write_json(os.path.join(out, "request_example.json"),
               build_payload(args.model or "<model from the endpoints file>", prompt,
                             user_text, decoding, args.extra, guided=guided,
                             drop=args.drop, stream=args.stream))
    if args.dry_run:
        # No run.json here. It pins the prompt and the decoding for every later session,
        # and a dry run sends nothing, so there is no results.jsonl for it to protect.
        # Writing it would lock the directory to a prompt you were only trying out, and
        # the next --dry-run with a different prompt would be refused.
        log(f"dry run: no calls, and no run.json. Wrote order.csv, prompt_example.txt, "
            f"request_example.json in {rel(out)}")
        return EXIT_OK

    check_run_json(out, meta, session)

    if (args.base_url or args.model) and (args.endpoint or
                                          args.endpoints != ENDPOINTS_DEFAULT):
        raise ConfigError("--base-url/--model describe one endpoint instead of an "
                          "endpoints file; giving both would silently ignore the file")
    if args.base_url or args.model:
        ep_cfgs = cli_endpoint(args)
    else:
        ep_cfgs = load_endpoints(resolve(args.endpoints), args.endpoint)
    if args.mode == "shard" and len({e["model"] for e in ep_cfgs}) > 1:
        raise ConfigError(f"shard mode needs one model on every endpoint, got "
                          f"{sorted({e['model'] for e in ep_cfgs})}; use --mode compare")

    latest, lines, torn = read_results(os.path.join(out, "results.jsonl"))
    if torn:
        log(f"results.jsonl: {torn} unusable line(s) ignored; those rows will be re-sent")
    skip = {k for k, r in latest.items()
            if not (args.retry_errors and (r.get("error") or r.get("parse_error")))}
    selected = items[:args.limit] if args.limit else items

    def key_for(it: Item, model: str, endpoint: str = "") -> str:
        return row_key(it.id, model, prompt, endpoint, guided)

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

    run = Run(args, prompt, decoding, meta, out, by_ep, args.mode == "shard",
              ep_cfgs, done=skip & planned_keys, canary_item=items[0])
    run.planned = planned_keys

    pid_path = os.path.join(out, "judge.pid")
    with open(pid_path, "w") as f:
        f.write(f"{os.getpid()}\n")
    try:
        return run_async(run.main())
    finally:
        with open(pid_path) as f:
            if f.read().strip() == str(os.getpid()):
                os.remove(pid_path)


if __name__ == "__main__":
    raise SystemExit(main())
