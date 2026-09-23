"""Judge data_selected_2/colab (real + CTGAN + TVAE rows) with the OpenAI API.

    export GPT_API_KEY=...
    python exp_scripts/exp_03.py --dry-run
    python exp_scripts/exp_03.py --dataset liver_ilpd
    python exp_scripts/exp_03.py --run-id <id> --retry-errors --max-tokens 8192   # truncated rows
    python exp_scripts/exp_03.py --run-id <id> --limit 5000                       # grow the sample

Pass ``--run-id <timestamp>`` to continue a previous run.

Each dataset directory is judged with its own prompt (PROMPTS below). Model and request
settings are exp_01's: gpt-5-mini, reasoning_effort medium, 4096 completion tokens, at
most 1000 rows / 1200 requests per file (a random sample where a file is larger).
``--limit`` takes the first N of a fixed shuffled order, so raising it on a resumed run
judges only the rows past the old limit.
"""

import argparse
import csv
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmjudge import judge  # noqa: E402


INPUTS = ROOT.parent / "data_selected_2/colab"
RESULTS = ROOT / "exp_results/exp_03"
PROMPTS = {"cervical_cancer": "c-cervical", "diabetes130": "c-diabetes",
           "diabetes130_big": "c-diabetes", "liver_ilpd": "c-liver"}


def n_rows(path: Path) -> int:
    with open(path, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", help="UTC timestamp printed by an earlier run")
    parser.add_argument("--dataset", action="append", choices=sorted(PROMPTS),
                        help="repeatable; default all")
    parser.add_argument("--limit", type=int, default=1000, metavar="N",
                        help="rows per file (default 1000, exp_01's)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="completion budget (default 4096, exp_01's); raise it with "
                             "--retry-errors to recover rows cut off at the budget")
    parser.add_argument("--retry-errors", action="store_true",
                        help="re-send rows whose latest record is an error")
    args = parser.parse_args(argv)

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if run_id in (".", "..") or Path(run_id).name != run_id:
        parser.error("--run-id must be a directory name, not a path")

    api_key = os.environ.get("GPT_API_KEY") or os.environ.get("LLMJUDGE_API_KEY")
    if not api_key and not args.dry_run:
        parser.error("set GPT_API_KEY first")

    jobs = [(d, p) for d in sorted(args.dataset or PROMPTS)
            for p in sorted((INPUTS / d).glob("*.csv"))]
    total = sum(min(n_rows(p), args.limit) for _, p in jobs)
    out_root = RESULTS / run_id
    print(f"experiment output: {out_root}")
    print(f"{len(jobs)} files, {total} rows to judge")

    for dataset, path in jobs:
        print(f"input: {dataset}/{path.name} (prompt {PROMPTS[dataset]})")
        code = judge(
            items=str(path),
            out=str(out_root / dataset / path.stem),
            run_tag=f"exp_03-{run_id}-{dataset}-{path.stem}",
            prompt=PROMPTS[dataset],
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            model="gpt-5-mini",
            guided="on",
            max_tokens=args.max_tokens,
            extra={"reasoning_effort": "medium", "max_completion_tokens": args.max_tokens},
            drop=["temperature", "max_tokens"],
            timeout=300,
            limit=args.limit,
            max_requests=args.limit + 200,    # 1200 at exp_01's limit
            retry_errors=args.retry_errors,
            dry_run=args.dry_run,
        )
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
