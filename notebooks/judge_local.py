"""
The minimal shape of the contract, kept as the readable reference: items.jsonl in,
verdicts.jsonl out, keyed on "id", resumable by re-running. `llmjudge/run.py` is the
production version of this loop -- endpoint pool, circuit breakers, adaptive window --
and step 2 of the plan makes it read and write exactly these two files.

Run this on YOUR machine, not in Colab.

    pip install openai
    export MEDGEMMA_BASE_URL="https://<random>.trycloudflare.com/v1"
    export MEDGEMMA_API_KEY="<the key notebooks/serve_vllm.py printed>"
    python notebooks/judge_local.py

Resumable: re-run after a disconnect and it skips finished items.
"""

import os
import json
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

BASE_URL = os.environ["MEDGEMMA_BASE_URL"]
API_KEY = os.environ["MEDGEMMA_API_KEY"]

MODEL = "medgemma-27b-it"

INPUT_PATH = "items.jsonl"      # one JSON object per line, must have "id"
OUTPUT_PATH = "verdicts.jsonl"

# Must not exceed --max-num-seqs on the server.
# Higher just queues requests server-side, and queue time
# counts against Cloudflare's 100s timeout.
CONCURRENCY = 16

# Keep bounded so a single call finishes well under 100s.
MAX_TOKENS = 512

MAX_RETRIES = 6


client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=90.0,
    max_retries=0,          # we handle retries ourselves
)

write_lock = threading.Lock()


# ============================================================
# PROMPT — replace with your actual judge rubric
# ============================================================

SYSTEM_PROMPT = (
    "You are a careful evaluator. Respond with JSON only: "
    '{"score": <1-5>, "reason": "<one sentence>"}. '
    "No markdown, no preamble."
)


def build_user_prompt(item):
    return (
        f"Question:\n{item['question']}\n\n"
        f"Answer:\n{item['answer']}\n\n"
        "Score the answer."
    )


# ============================================================
# ONE CALL, WITH RETRY
# ============================================================

def judge(item):

    for attempt in range(MAX_RETRIES):

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(item)},
                ],
                temperature=0.0,
                max_tokens=MAX_TOKENS,
            )

            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()

            return {"id": item["id"], "verdict": json.loads(text)}

        except Exception as e:

            if attempt == MAX_RETRIES - 1:
                return {"id": item["id"], "error": repr(e)}

            # Exponential backoff with jitter.
            # Covers 429 (tunnel concurrency cap), 524
            # (edge timeout), and dropped connections.
            time.sleep((2 ** attempt) + random.random())


def judge_and_write(item):

    result = judge(item)

    with write_lock:
        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(result) + "\n")
            f.flush()

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    items = [json.loads(line) for line in open(INPUT_PATH)]

    done = set()

    if os.path.exists(OUTPUT_PATH):
        for line in open(OUTPUT_PATH):
            try:
                row = json.loads(line)
                if "error" not in row:
                    done.add(row["id"])
            except Exception:
                pass

    todo = [i for i in items if i["id"] not in done]

    print(f"{len(done)} done, {len(todo)} remaining")

    started = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:

        for _ in pool.map(judge_and_write, todo):

            completed += 1

            if completed % 50 == 0:
                rate = completed / (time.time() - started)
                eta = (len(todo) - completed) / rate / 60
                print(
                    f"{completed}/{len(todo)}  "
                    f"{rate:.1f} req/s  ETA {eta:.0f} min"
                )

    errors = sum(
        1 for line in open(OUTPUT_PATH)
        if "error" in json.loads(line)
    )

    print(f"\nDone. {errors} errors — re-run to retry them.")


if __name__ == "__main__":
    main()
