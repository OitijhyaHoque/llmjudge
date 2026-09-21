# llmjudge — an LLM judge, and nothing else

This repository sends rows to a model and records what it answers. It holds **no rules,
no cards, no knowledge tables, no dataset-specific column names, and no opinion about what
the model should answer** — the prompt decides that. Those live elsewhere:

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

From a checkout:

```bash
python3 -m pip install -e .                           # or: pip install -r requirements.txt
cp .env.example .env                                  # fill in the URL and key
cp configs/endpoints.example.toml configs/endpoints.toml

python3 -m unittest discover -s tests -t .            # 51 tests, ~33s, no server needed
llmjudge --items items.jsonl --out out/pilot \
    --prompt c3 --run-tag pilot-01 --dry-run
```

From anywhere else — a Colab notebook, a colleague's machine — install the tag instead,
and nothing is checked out at all:

```bash
pip install "git+https://$GH_TOKEN@github.com/<org>/llmjudge.git@v0.1.0"
llmjudge --items /content/drive/MyDrive/judge/items.jsonl --out results/pilot \
    --prompt c3 --run-tag pilot-01 \
    --base-url https://host/v1 --model medgemma-27b-it
```

The prompts ship inside the package, so `--prompt c3` works with no checkout. Every other
path — items, out, endpoints, `.env` — is yours, and is read relative to the directory you
run in. `python3 -m llmjudge` is the same entry point as the `llmjudge` command.

`--dry-run` renders the first prompt, prints it, and sends nothing. Always the first step
against a new prompt or a new endpoint. It writes no `run.json`, so you can dry-run one
prompt after another into the same directory and compare them.

## Saying where the model is

One server needs no files at all:

```bash
export LLMJUDGE_API_KEY=...                  # or --api-key, but that lands in your history
llmjudge --base-url https://host/v1 --model gpt-4o-mini ...
```

`--base-url` also comes from `LLMJUDGE_BASE_URL`, so a Colab cell that already exported
the URL of the server it just started needs neither flag written down.

Several servers — two notebooks sharing a pool, or MedGemma against a frontier model in
`--mode compare` — go in an endpoints file, one entry each: `cp
configs/endpoints.example.toml configs/endpoints.toml`. An entry names the *environment
variables* holding its URL and key, never the values. Those are read from the entry's
`env_file` if there is one, and from the process environment otherwise.

### Slow replies, and a ceiling on the bill

```bash
llmjudge --stream                 # read the reply as it is written
llmjudge --max-requests 2000      # and stop the run there
```

Cloudflare cuts a request whose origin has sent nothing for about 100 s, which is why
`--timeout` defaults to 95. A prompt that reasons before it answers can spend longer than
that on a reply the server is still buffering, and the tunnel kills a row the model is
still working on. `--stream` reads the reply as server-sent events, so the timeout measures
*silence* instead of the whole reply and `--timeout` can go well past the tunnel's limit.
The assembled answer, its token counts and its finish reason are the same either way. Off
by default; `stream = true` sets it for one endpoint.

`--max-requests` stops the run after N requests. It is the only thing between a typo and
55,000 rows against a frontier model, so it counts everything the run sends: rows,
preflight and probes. The stop is the Ctrl-C stop — requests in flight finish, the summary
is written, and the answers already on disk stay, so a re-run picks up where it left off.

### When the API is not shaped like vLLM

The base URL is used as you write it. A URL that already carries a path keeps that path,
so Azure's deployment URL and a gateway's prefix work; only a bare host gets `/v1`
appended, which is what every vLLM URL looked like before. A query string stays at the
end, where Azure wants its `api-version`.

Three more knobs move the rest of the request, and each also works per entry in the
endpoints file (`chat_path`, `auth_header`, `headers`, `models_path`):

```bash
llmjudge --chat-path /responses                 # when it is not /chat/completions
         --auth-header api-key                  # the key, raw, in another header
         --header 'x-portkey-provider: azure'   # repeatable; an empty value removes one
         --skip-model-check                     # this API lists no models
```

A key sent through a header other than `Authorization` goes raw, with no `Bearer` — that
is what Azure wants. `--header` applies on top of every endpoint, so `--header
'ngrok-skip-browser-warning:'` drops a default the judge would otherwise send.

`--skip-model-check` turns off only the question *is this model served*. The canary still
has to come back sound, so an endpoint is still proven before a single row is sent.

Azure needs all four at once:

```bash
llmjudge --base-url 'https://R.openai.azure.com/openai/deployments/D?api-version=2024-06-01' \
    --model gpt-4o-mini --auth-header api-key --skip-model-check --drop seed ...
```

What none of this changes is the **body**: a `messages` array with the system prompt as its
first message, and `response_format` for guided decoding. An API with a body of its own —
Anthropic's native `/v1/messages`, where the system prompt is a top-level field and there
is no `response_format` — needs an adapter, which is step 4 of the plan, not these flags.

### When the API does not take the same fields

Every request carries `model`, `messages`, `temperature`, `seed` and `max_tokens`.
Not every API accepts all of them: Anthropic has no `seed`, the reasoning models reject
`temperature`, and newer OpenAI models want `max_completion_tokens`. So:

```bash
llmjudge --drop seed --drop temperature ...                        # remove
llmjudge --extra '{"max_completion_tokens": 4096}' --drop max_tokens ...   # rename
```

`--extra` is merged into the body and `--drop` removes fields from it, so renaming a
field is a drop plus an add. A field set to `null` in `--extra` is removed too. Both are
visible in `<out>/request_example.json` under `--dry-run`, before anything is sent. An
endpoints file entry can carry its own `extra = { ... }` and `drop = [ ... ]`, and the
command line's apply on top of every endpoint.

## Your own prompt

The prompt is yours, and it is **required**: there is no default, because the prompt is the
only thing that says what is being judged. Four ways to give it, all equivalent once loaded
— what is pinned and resumed on is the **text**, never where it came from.

```bash
llmjudge --prompt c3 ...                             # one that ships with the package
llmjudge --prompt /content/drive/MyDrive/my-prompt   # a directory: system.md + user.md
llmjudge --system drive/sys.md --user drive/row.md --prompt-name tickets   # any two files
```

```python
from llmjudge import judge                           # in a notebook: the text itself

SYSTEM = """You check support tickets for contradictions.
Answer with this JSON object and nothing else:
{"verdict": "consistent" | "inconsistent" | "unsure", "short_reason": "<15 words>"}
"""
USER = """Ticket {ticket_id}
product: {product}
opened:  {opened}
closed:  {closed}
"""

judge(items="/content/drive/MyDrive/judge/items.jsonl",
      out="/content/drive/MyDrive/judge/results/tickets",
      run_tag="tickets-01", system=SYSTEM, user=USER, guided="off", limit=100)
```

`judge()` takes every command-line option with underscores (`max_tokens`, `retry_errors`,
`dry_run`, `base_url`, `model`), a dict option as JSON (`extra={"top_p": 0.9}`) and a list
option as a repeated flag (`drop=["seed"]`). It works inside a Colab or Jupyter cell,
which runs in an event loop of its own. `system=` and `user=` are the prompt text, or the path to a file holding it —
a value that names an existing file is read, anything else is the prompt. The text is
copied to `<out>/prompt/`, so a results directory always carries the prompt that produced
it, and re-running the same cell resumes rather than starts again.

### The answer shape comes from your prompt

One rule, and it is the only one: the system prompt shows the **answer as a JSON example**,
on its own line or lines, starting with `{`. That example *is* the contract — the judge
reads the keys, their order, and the values each may take out of it, and knows nothing else
about your answer.

```
{"score": 1 | 2 | 3 | 4 | 5, "why": "<one sentence>"}
{"verdict": "consistent" | "inconsistent" | "unsure", "short_reason": "<25 words>"}
{"verdict": "pass" | "fail", "findings": [{"severity": "high" | "low", "note": "<why>"}]}
```

- `"a" | "b" | "c"` is a choice, and becomes an enum in the JSON schema sent as
  `response_format`. Numbers work: `1 | 2 | 3`.
- `"<anything in angle brackets>"` is free text.
- Nesting works: objects, and arrays whose one shown element describes the rest.
- The key order is the schema's order, so a prompt that tells the model to reason before
  answering actually gets that under guided decoding.
- **A key that offers a choice is the label**: the thing `summary.json` counts, the
  agreement check compares in `--mode compare`, and the canary pins when it checks that
  your server really enforces `response_format`. It can be called anything, and the first
  one wins if there are several. A prompt need not have one — an answer that is prose, or
  a free number, is recorded whole like any other; there is then nothing to count, so the
  summary reports how many rows were answered and leaves the rates out, and the canary
  pins the first key instead.

**Showing the model other examples.** Few-shot examples are between your prompt and the
model, and the judge has no business reading them — but it cannot tell which object is the
contract either. So when the system prompt holds more than one JSON object, fence the one
that is the contract, and write as many others as you like:

````
Here is a good answer:
{"verdict": "consistent", "short_reason": "no conflict"}
...and a bad one:
{"verdict": "consistent", "short_reason": "the patient is male and 40 weeks pregnant"}

Answer with this object and nothing else:
```answer
{"verdict": "consistent" | "inconsistent" | "unsure", "short_reason": "<25 words>"}
```
````

The user template is free too: any `{column}` in it is filled from the item's `fields`,
and every row is checked for every column before the run starts.

## The interface

**In** — one JSONL, one object per row:

```json
{"id": "ctgan_split:41772", "fields": {"age": "[70-80)", "diag_1": "250.83"},
 "group": "kept", "stratum": "accept/label=1", "weight": 2399.75}
```

| key | | |
|---|---|---|
| `id` | required | unique; the only thing tying a verdict back to a row, and never parsed here |
| `fields` | required | the `{column}` substitutions for the user template; numbers and strings both |
| `group`, `stratum` | optional | opaque labels that bucket `summary.json`; nothing here reads their meaning |
| `weight` | optional, 1.0 | for the weighted rates, so a caller that sampled strata unequally can still report a population rate |

**Out** — `<out>/results.jsonl`, append-only, one object per final answer: `id`, `answer`
(the model's object, whatever shape your prompt asked for), `verdict` (the label's value,
under a fixed name so a reader need not know what you called it), `reasoning`, the endpoint
and model, the prompt and schema shas, latency and token counts, and `error` / `parse_error`
where there is one. Errors are recorded, never
dropped, so a re-run retries exactly them.

Building the items file is the caller's job, deliberately: stratifying a pool needs to know
what a rule decided, and this repository holds no rules.

## Layout

```
llmjudge/
├── llmjudge/
│   ├── run.py        the judge: endpoint pool, retries, breakers, resume, summary
│   ├── template.py   {column} substitution, and the refusal when a column is missing
│   ├── prompts/<name>/  system.md + user.md — c1, c1r, c2, c2-reason-first, c3,
│   │                 c3-reasoning, c3-reasoning-2. Inside the package so that pip
│   │                 install ships them; your own prompt need not live here at all.
│   └── __main__.py   python3 -m llmjudge
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

**A pool that grew.** The resume key is the row's `id` with the model, prompt and schema —
not the items file. Append 2,000 rows to the file, re-run into the same directory, and
only the new ids are sent. `run.json` restates which pool each session ran.

**Refusal on a changed run.** The first real run writes `<out>/run.json`, which pins the
prompt, the schema, the sampling settings, the send-order seed and the mode. A restart into that directory with any of them
changed is refused rather than quietly mixing two kinds of answer in one file.

`--max-tokens` is the one setting you may change. A reply cut off at the budget is recorded
as an error and never as a verdict, so no answer on disk was shaped by the old budget:
raise it, re-run with `--retry-errors`, and exactly the truncated rows are sent again. Each
session's budget is kept in `run.json`'s `sessions`.

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
out — ids present and unique, `fields` an object, and **every row** carrying every
`{column}` the prompt names — so a pool malformed on line 40,000 costs nothing rather than
four hours. Field values may be numbers or strings; a number is rendered as it reads. Then,
per endpoint: the model listing must name the configured model — unless
`--skip-model-check` says this API has no such listing — and a canary request must come
back sound. Under guided decoding the canary's schema admits one value,
so a server silently ignoring `response_format` is refused rather than producing a run of
unconstrained replies.

**No row can hang the run.** Anything raised while rendering or sending a row is recorded
against that row as an `internal` error and the slot is returned, so one unrenderable row
costs one row rather than stalling a pool that can never finish.

**Nothing is lost quietly.** A truncated reply (`finish_reason=length`) is recorded as an
error, never as a verdict. Non-JSON or wrong-key replies are retried once, then recorded
with `parse_error`. Exit codes: `0` every planned row recorded, `2` configuration refused,
`3` stopped incomplete. Ctrl-C once drains the in-flight requests and writes the summary;
twice exits immediately, and every recorded line is already on disk.

## Secrets

Never in the source, never in a notebook body.

| where it runs | how it gets the key |
|---|---|
| one server, anywhere | `LLMJUDGE_API_KEY` in the environment, read when `--api-key` is not given |
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
