# llmjudge — an LLM judge, and nothing else

This repository sends rows to a model and records verdicts. It holds **no rules, no
cards, no knowledge tables and no dataset-specific column names**. Those live elsewhere:

```
rule-builder/   authors card.json + knowledge/        writes
judge-0/        runs the rules, builds the pool,      writes items  ─┐
                evaluates the filtered corpus         reads verdicts ─┤
llmjudge/       sends rows to a model                                ─┘
```

The line between them is **files**, never an import. Nothing here reads a `card.json` or
a `decisions.csv`, so a colleague with a different dataset — or no rules at all — can use
it unchanged.

## Quickstart

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env                                  # fill in the URL and key
cp configs/endpoints.example.toml configs/endpoints.toml

python3 -m unittest discover -s tests -t .            # 21 tests, ~18s, no server needed
python3 -m llmjudge --items items.jsonl --out out/pilot \
    --prompt c3 --run-tag pilot-01 --dry-run
```

`--dry-run` renders the first prompt, prints it, and sends nothing. Always the first step
against a new prompt or a new endpoint.

## The interface

**In** — one JSONL, one object per row:

```json
{"id": "ctgan_split:41772", "fields": {"age": "[70-80)", "diag_1": "250.83"},
 "group": "kept", "stratum": "accept/label=1", "weight": 2399.75}
```

| key | | |
|---|---|---|
| `id` | required | unique; the only thing tying a verdict back to a row, and never parsed here |
| `fields` | required | the `{column}` substitutions for `prompts/<name>/user.md` |
| `group`, `stratum` | optional | opaque labels that bucket `summary.json`; nothing here reads their meaning |
| `weight` | optional, 1.0 | for the weighted rates, so a caller that sampled strata unequally can still report a population rate |

**Out** — `<out>/results.jsonl`, append-only, one object per final answer: `id`, `verdict`,
`short_reason`, `reasoning`, the endpoint and model, the prompt and schema shas, latency and
token counts, and `error` / `parse_error` where there is one. Errors are recorded, never
dropped, so a re-run retries exactly them.

Building the items file is the caller's job, deliberately: stratifying a pool needs to know
what a rule decided, and this repository holds no rules.

## Layout

```
llmjudge/
├── llmjudge/
│   ├── run.py        the judge: endpoint pool, retries, breakers, resume, summary
│   ├── prompts.py    {column} substitution, and the refusal when a column is missing
│   └── __main__.py   python3 -m llmjudge
├── prompts/<name>/   system.md + user.md — c1, c1r, c2, c2-reason-first, c3,
│                     c3-reasoning, c3-reasoning-2
├── configs/endpoints.example.toml    names of env keys, never values
├── notebooks/
│   ├── serve_vllm.py    the Colab cell that serves MedGemma behind a Cloudflare tunnel
│   └── judge_local.py   the readable reference loop: items.jsonl → verdicts.jsonl
└── tests/            mock_vllm.py + test_run.py against a threaded fake server
```

## What the judge guarantees

Everything below is already implemented and covered by `tests/test_run.py`.

**Resume.** `<out>/results.jsonl` is append-only, flushed per line and fsync'd every 50.
A restart skips every row already recorded. `--retry-errors` re-sends only the rows whose
latest record is an error.

**Refusal on a changed run.** `<out>/run.json` pins the prompt, the schema, the decoding
settings, the send-order seed and the items file's sha. A restart into that directory with
any of them changed is refused rather than quietly mixing two pools in one file.

**Endpoints that come and go.** Connection errors, `ERR_NGROK_*`, 502/503/504 and
Cloudflare 520–527 pause an endpoint; `/v1/models` and a canary are re-probed with
backoff and it rejoins on its own. Five consecutive timeouts also pause it, because a
dead engine still answers `/v1/models`. While an endpoint is down its env file is re-read,
so a Colab restart with a new `trycloudflare` hostname is picked up without stopping the
run. In shard mode the other endpoints drain the queue meanwhile.

**Backpressure.** The in-flight window grows while measured throughput rises, steps back
when a step buys nothing, and halves on 429 or timeout. Bounds come from the endpoints
file; keep `max_concurrency` at or below vLLM's `--max-num-seqs`.

**Preflight.** The whole items file is parsed and checked before a single request goes
out — ids present and unique, `fields` an object, and the prompt's `{column}` placeholders
all present — so a pool malformed on line 40,000 costs nothing rather than four hours. Then,
per endpoint: `/v1/models` must list the configured model, and a canary request must come
back sound. Under guided decoding the canary's schema admits one value,
so a server silently ignoring `response_format` is refused rather than producing a run of
unconstrained replies.

**Nothing is lost quietly.** A truncated reply (`finish_reason=length`) is recorded as an
error, never as a verdict. Non-JSON or wrong-key replies are retried once, then recorded
with `parse_error`. Exit codes: `0` every planned row recorded, `2` configuration refused,
`3` stopped incomplete. Ctrl-C once drains the in-flight requests and writes the summary;
twice exits immediately, and every recorded line is already on disk.

## Secrets

Never in the source, never in a notebook body.

| where it runs | how it gets the key |
|---|---|
| your machine | `.env`, gitignored; the endpoints file names the key, not its value |
| Colab | **Colab Secrets** (key icon in the sidebar), read with `userdata.get` |
| the served endpoint's own key | generated fresh per session by `notebooks/serve_vllm.py` |

The vLLM API key is no longer a constant. The tunnel hostname is random per session
anyway, so a fixed key bought nothing and could only leak.

## What is still to do

Steps 1 and 2 of `notes/20260921_044121-judge-plan.md` are done: the extraction, and the
items interface. What the plan changes next:

1. **Nobody writes the items file yet.** Step 3 moves pool construction and stratification
   into `judge-0/pipeline/judge_stage.py`, which writes items and reads verdicts. Until
   then a caller emits the JSONL itself.
2. **One provider shape.** Step 4 adds `providers.py`: the OpenAI-compatible path stays the
   default, and an Anthropic adapter joins it for the frontier-model comparison.
3. **The Colab path is not here yet.** Step 5 adds `llmjudge/colab.py`: mount Drive, pull
   any partial results down to resume, run against localhost, mirror the tail back every
   100 s, verify the copy at exit. No bundles and no zips.

Runs made before the items interface keyed resume on `(table sha, arm, row index)`. They do
not resume here, by decision; their `results.jsonl` stays readable.
