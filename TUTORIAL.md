# Using `llmjudge`: a step-by-step tutorial

`llmjudge` sends one row per request to an OpenAI-compatible chat-completions server and
writes the model's answers to disk. The rows are a CSV or a JSONL file.

```text
rows.csv + prompt + model endpoint
                  |
                  v
              llmjudge
                  |
                  v
       results.jsonl + summary.json
```

The model can run on the same computer, on a DGX Spark, or inside a Colab runtime.
Only the endpoint URL changes; the items and prompt have the same format everywhere.

## 1. Install `llmjudge`

`llmjudge` requires Python 3.11 or newer. From this repository:

```bash
cd llmjudge
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
llmjudge --help
```

To install a tagged version from GitHub instead:

```bash
python3 -m pip install "git+https://github.com/<org>/llmjudge.git@v0.1.0"
```

For a private repository, authenticate with a read-only GitHub token. Do not put the
token in a notebook cell or commit it to a file.

## 2. Prepare the three inputs

A run needs:

1. the rows to judge, as a CSV or an items JSONL file;
2. a system prompt and user template;
3. an OpenAI-compatible model endpoint.

### 2.1 Point at the rows to judge

If the rows are already a CSV, pass it to `--items` directly. Every column becomes a
field the user template can name:

```text
question,candidate_answer
What is 2 + 2?,4
What is the capital of France?,London
```

```bash
--items rows.csv
```

Four column names are reserved: `id`, `group`, `stratum` and `weight` are read as the
item's keys rather than as fields, exactly as the JSONL keys below. Without an `id`
column the row number is the id, which is all that resume needs.

A CSV covers most runs. Write JSONL instead when a field is nested (an array or an
object), or when the pool was sampled by stratum and carries weights.

#### The JSONL form

The input is JSON Lines: one JSON object per line. Every object must have a unique `id`
and a `fields` object.

```json
{"id":"example-1","fields":{"question":"What is 2 + 2?","candidate_answer":"4"},"group":"smoke"}
{"id":"example-2","fields":{"question":"What is the capital of France?","candidate_answer":"London"},"group":"smoke"}
```

Save those two lines as `items.jsonl`. The supported keys are:

| Key | Required | Meaning |
|---|---:|---|
| `id` | yes | Unique row identifier. It ties the result back to the input. |
| `fields` | yes | Values inserted into the user template. Values may be strings, numbers, arrays, or objects. |
| `group` | no | Opaque label used to split `summary.json`. |
| `stratum` | no | Finer opaque label reported within a group. |
| `weight` | no | Sampling weight used for weighted rates; defaults to `1.0`. |

Only `fields` is shown to the model. The model does not see `id`, `group`, `stratum`, or
`weight`.

To draw a stratified sample from a large CSV rather than judging all of it,
`llmjudge make-items` writes the JSONL using a TOML pool spec:

```bash
llmjudge make-items \
  --spec configs/pool.diabetes130.toml \
  --pool pilot \
  --root /path/to/source/runs \
  --out items.jsonl
```

The included `configs/pool.diabetes130.toml` is an example. This is for sampling a pool
with per-stratum weights; to judge a whole CSV, pass it to `--items` and skip this.

### 2.2 Create the prompt

Create `prompts/demo/system.md`:

```text
You judge whether a candidate answer correctly answers the question.
Answer with this JSON object and nothing else:
{"verdict": "pass" | "fail", "reason": "<one short sentence>"}
```

Create `prompts/demo/user.md`:

```text
Question: {question}
Candidate answer: {candidate_answer}
```

Every `{name}` in the user template must exist in every item's `fields`. `llmjudge`
checks the entire items file before sending any request.

The JSON example in the system prompt is the answer contract. It defines the output
keys, their order, and allowed label values. Here, `verdict` is the label summarized as
`pass` or `fail`.

You can pass this prompt in either form:

```bash
# Directory containing system.md and user.md
--prompt prompts/demo

# Two explicit files
--system prompts/demo/system.md --user prompts/demo/user.md --prompt-name demo
```

Packaged prompts such as `c1`, `c2`, and `c3` are also accepted with
`--prompt c3`, but their user templates expect the dataset columns used by those
examples.

### 2.3 Identify the endpoint

`llmjudge` does not load model weights itself. It calls a server that provides:

- `GET /v1/models`;
- `POST /v1/chat/completions`;
- OpenAI-style `messages` request bodies.

The normal single-server arguments are:

```bash
--base-url http://127.0.0.1:8000/v1 --model local-judge
```

The value of `--model` must match a model returned by `/v1/models`. A local server may
need no key. When it does, prefer an environment variable over `--api-key`:

```bash
export LLMJUDGE_API_KEY='<token>'
```

## 3. Run with a model on the local computer

This example uses vLLM because it exposes the API `llmjudge` expects. vLLM's current
[quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/) lists supported
installation and hardware options.

### Step 1: start the model server

In terminal 1:

```bash
python3 -m pip install vllm

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name local-judge \
  --generation-config vllm
```

Replace the model handle with a chat model supported by the installed vLLM version and
your hardware. If a compatible local server is already running, skip this step.

### Step 2: verify the server

In terminal 2, from the `llmjudge` repository:

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

Confirm that the response lists `local-judge`.

### Step 3: dry-run one request

```bash
llmjudge \
  --items rows.csv \
  --out out/demo-local \
  --system prompts/demo/system.md \
  --user prompts/demo/user.md \
  --prompt-name demo \
  --run-tag demo-local-01 \
  --base-url http://127.0.0.1:8000/v1 \
  --model local-judge \
  --dry-run
```

The dry run sends nothing. Inspect the rendered prompt and request:

```bash
sed -n '1,160p' out/demo-local/prompt_example.txt
python3 -m json.tool out/demo-local/request_example.json
```

### Step 4: run the judge

Remove `--dry-run`:

```bash
llmjudge \
  --items rows.csv \
  --out out/demo-local \
  --system prompts/demo/system.md \
  --user prompts/demo/user.md \
  --prompt-name demo \
  --run-tag demo-local-01 \
  --base-url http://127.0.0.1:8000/v1 \
  --model local-judge
```

Guided decoding is on by default: the prompt's JSON contract is sent as
`response_format`. If an otherwise compatible server does not implement structured
output, add `--guided off`. The model must then return a matching JSON object itself;
`llmjudge` extracts the last matching object from the reply.

Re-running the same command resumes the run. Completed rows are not sent again. To retry
only rows whose latest record is an error or parse error, add `--retry-errors`.

## 4. Run on a DGX Spark

The simplest layout runs both vLLM and `llmjudge` on the Spark, with the API bound only
to localhost. NVIDIA maintains the current model matrix, image guidance, and Spark
prerequisites in its official
[vLLM for DGX Spark playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md).
Use a model and container combination listed there; Spark is ARM64/Blackwell, so a
generic x86 CUDA image is not a safe assumption.

### Step 1: check the Spark prerequisites

```bash
nvidia-smi
docker --version
docker ps
```

Docker must have access to the NVIDIA GPU. Obtain a Hugging Face token if the selected
model requires one.

### Step 2: choose a tested image and model

On the Spark:

```bash
export HF_TOKEN='<hugging-face-token>'
export VLLM_IMAGE='nvcr.io/nvidia/vllm:<tested-tag>'
export MODEL_HANDLE='nvidia/Qwen3-8B-NVFP4'
export SERVED_MODEL='spark-judge'

docker pull "$VLLM_IMAGE"
```

Replace `<tested-tag>` with a version from NVIDIA's current NGC vLLM instructions. If
the model's card or the Spark playbook gives model-specific flags or a different image,
use that recipe.

### Step 3: start vLLM in Docker

```bash
docker run --rm --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  "$VLLM_IMAGE" \
  vllm serve "$MODEL_HANDLE" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "$SERVED_MODEL" \
  --max-num-seqs 8
```

Model loading can take several minutes. In another Spark terminal:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

`127.0.0.1:8000:8000` keeps the endpoint private to the Spark. To call it from another
computer, bind it to a reachable interface and protect that network path; do not expose
an unauthenticated vLLM server directly to the internet.

### Step 4: install `llmjudge` and copy the inputs

Clone this repository on the Spark and install it as shown in section 1. Copy
`rows.csv` and `prompts/demo/` to the checkout, or use absolute paths to files already
on the Spark.

### Step 5: dry-run, then run

For a small run, the single-endpoint command is enough:

```bash
llmjudge \
  --items rows.csv \
  --out out/demo-spark \
  --prompt prompts/demo \
  --run-tag demo-spark-01 \
  --base-url http://127.0.0.1:8000/v1 \
  --model spark-judge \
  --dry-run

llmjudge \
  --items rows.csv \
  --out out/demo-spark \
  --prompt prompts/demo \
  --run-tag demo-spark-01 \
  --base-url http://127.0.0.1:8000/v1 \
  --model spark-judge
```

For a long run, cap the judge's concurrency at the server's `--max-num-seqs`. Create
`configs/endpoints.spark.toml`:

```toml
[[endpoint]]
name = "spark"
url_key = "SPARK_BASE_URL"
model = "spark-judge"
concurrency = 4
max_concurrency = 8
metrics = true
```

Then run:

```bash
export SPARK_BASE_URL='http://127.0.0.1:8000/v1'

llmjudge \
  --items rows.csv \
  --out out/demo-spark \
  --prompt prompts/demo \
  --run-tag demo-spark-01 \
  --endpoints configs/endpoints.spark.toml
```

If `--max-num-seqs` changes, keep `max_concurrency` at or below it.

## 5. Run in the included Colab notebook

The notebook path is `notebooks/run_judge.ipynb`. It installs `llmjudge`, mounts Drive,
starts the packaged MedGemma/vLLM server, runs the judge, mirrors results to Drive, and
prints the summary.

The notebook currently contains `REPO = 'your-org/llmjudge'`. It cannot install the
package until the repository has a GitHub remote and the chosen tag exists.

### Step 1: prepare GitHub and Drive

1. Push the repository and create the tag used by the notebook, such as `v0.1.0`.
2. Upload `rows.csv` to `MyDrive/judge/rows.csv`.
3. Open `notebooks/run_judge.ipynb` in Colab.
4. Select **Runtime -> Change runtime type -> A100 or L4**.

### Step 2: add Colab Secrets

In the Secrets panel (key icon), add and enable notebook access for:

- `GH_TOKEN`: a fine-grained, contents-read-only token for a private repository;
- `HF_TOKEN`: a token allowed to download the selected Hugging Face model.

Do not paste either token into a code cell.

### Step 3: edit the notebook configuration

In the first code cell, set:

```python
REPO = "<org>/llmjudge"
TAG = "v0.1.0"
DRIVE = "/content/drive/MyDrive/judge"
```

In the pool cell, point `ITEMS` at the uploaded file:

```python
ITEMS = f"{DRIVE}/rows.csv"
```

The packaged serving cell exports `LLMJUDGE_BASE_URL`, `LLMJUDGE_API_KEY`, and
`LLMJUDGE_MODEL`, so the judging cell does not need endpoint arguments.

### Step 4: select the prompt and output

The included judging cell uses a packaged prompt:

```python
from llmjudge.colab import run

exit_code = run(
    items=ITEMS,
    out=f"{DRIVE}/results/pilot",
    run_tag="pilot-01",
    prompt="c3-reasoning-2",
    guided=False,
    max_tokens=2560,
)
print("exit", exit_code)
```

For the demo prompt created earlier, upload its files to Drive and pass their text:

```python
from pathlib import Path
from llmjudge.colab import run

prompt_dir = Path(DRIVE) / "prompts" / "demo"
exit_code = run(
    items=ITEMS,
    out=f"{DRIVE}/results/demo",
    run_tag="demo-colab-01",
    system=(prompt_dir / "system.md").read_text(),
    user=(prompt_dir / "user.md").read_text(),
)
print("exit", exit_code)
```

### Step 5: run the cells from top to bottom

The model download and server startup can take about ten minutes on the first run. The
judge writes to `/content` and mirrors to Drive every 100 seconds and at exit. Re-running
the judging cell restores the Drive copy and resumes incomplete work.

Exit code `4` means judging finished but the final Drive mirror is incomplete. The cell
prints the local runtime path containing the results; copy them before the runtime is
recycled.

## 6. Read the output

Every real run writes an output directory such as `out/demo-local/`:

| File | What it contains |
|---|---|
| `results.jsonl` | Append-only result records, including answers, errors, latency, and token counts. |
| `summary.json` | Latest-record counts, rates, error totals, and endpoint statistics. |
| `run.json` | The prompt, schema, item, decoding, and session metadata used for safe resume. |
| `endpoints.json` | Endpoint state, request counts, adaptive concurrency window, and recent events. |
| `order.csv` | Seeded order in which rows are sent. |
| `prompt_example.txt` | The rendered system and first user prompt. |
| `request_example.json` | The first request body, useful for debugging API compatibility. |
| `prompt/` | Prompt text copied into the result directory when passed through Python. |

### 6.1 Check whether the run completed cleanly

```bash
python3 -m json.tool out/demo-local/summary.json
```

The primary success checks are:

- `missing` is `0`;
- `parse_errors` is `0`;
- `errors` is `{}`;
- each endpoint's `truncated` count is `0`.

Exit codes have the same high-level meaning:

| Code | Meaning |
|---:|---|
| `0` | Every planned row has a final record. |
| `2` | Configuration was refused before the run. |
| `3` | The run stopped with rows still missing. Re-run to resume. |
| `4` | Colab only: final results were not fully mirrored to Drive. |

### 6.2 Read individual answers

`results.jsonl` contains one JSON object per recorded attempt. This standard-library
snippet prints the useful fields:

```bash
python3 - <<'PY'
import json

with open("out/demo-local/results.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        print(row["id"], row.get("answer"), row.get("error"), row.get("parse_error"))
PY
```

Important result fields include:

- `id`: the input ID;
- `answer`: the parsed object requested by the prompt;
- `verdict`: the value of the prompt's first choice-valued key, under a stable name;
- `reasoning`: text before the final JSON object in unguided mode;
- `raw`: the model's complete response text;
- `error` and `parse_error`: transport/model and answer-shape failures;
- `latency_s`, `prompt_tokens`, and `completion_tokens`: request measurements;
- `endpoint`, `model`, `prompt_sha`, and `schema_sha`: provenance.

Because the file is append-only, a retried row can have an older error followed by a
successful record. `summary.json` uses the latest record for each run key; raw line count
is therefore not the same as final row count.

### 6.3 Read rates by group

```bash
python3 - <<'PY'
import json

with open("out/demo-local/summary.json", encoding="utf-8") as f:
    summary = json.load(f)

print("rows:", summary["rows"])
print("missing:", summary["missing"])
print("errors:", summary["errors"])

for name, group in summary["by_group"].items():
    print(name, group["verdicts"])
    print("  rates:", group["rates"])
    print("  weighted:", group["rates_weighted"])
PY
```

`rates` are raw shares among answered rows. `rates_weighted` use each item's `weight`
and are the appropriate figures when the items were sampled unequally by stratum.

## 7. Common failures

- **Missing template field:** add the named field to every item's `fields`, or remove the
  placeholder from the user template. No request is sent until all rows validate.
- **Configured model is not served:** compare `--model` with `GET /v1/models`; for vLLM,
  set a stable name with `--served-model-name`.
- **Server ignores `response_format`:** use a server with structured-output support, or
  run with `--guided off` and require matching JSON in the prompt.
- **Truncated replies:** raise `--max-tokens`, then re-run with `--retry-errors`.
- **Changed prompt or sampling settings:** use a new output directory and run tag. A
  result directory is intentionally pinned so incompatible answers cannot be mixed.
- **Interrupted run:** run the same command again. `results.jsonl` is append-only and
  already completed rows are skipped.

