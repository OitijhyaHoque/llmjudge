"""Run the 20-row reason-first pilot with OpenAI GPT-5 Mini.

    export OPENAI_API_KEY=...
    python exp_scripts/exp_01.py --dry-run
    python exp_scripts/exp_01.py

Pass ``--run-id <timestamp>`` to continue a previous run.
"""

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmjudge import judge  # noqa: E402


INPUTS = ROOT / "exp_input/exp_01"
RESULTS = ROOT / "exp_results/exp_01"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", help="UTC timestamp printed by an earlier run")
    args = parser.parse_args(argv)

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if run_id in (".", "..") or Path(run_id).name != run_id:
        parser.error("--run-id must be a directory name, not a path")

    api_key = os.environ.get("GPT_API_KEY") or os.environ.get("LLMJUDGE_API_KEY")
    if not api_key and not args.dry_run:
        parser.error("set GPT_API_KEY first")

    items = sorted(INPUTS.glob("*.csv"))
    if not items:
        parser.error(f"no CSV files found in {INPUTS}")

    out = RESULTS / run_id
    print(f"experiment output: {out}")
    for path in items:
        print(f"input: {path.name}")
        code = judge(
            items=str(path),
            out=str(out / path.stem),
            run_tag=f"exp_01-{run_id}-{path.stem}",
            prompt="c3",
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            model="gpt-5-mini",
            guided="on",
            max_tokens=4096,
            extra={"reasoning_effort": "medium", "max_completion_tokens": 4096},
            drop=["temperature", "max_tokens"],
            timeout=300,
            limit=1000,
            max_requests=1200,
            dry_run=args.dry_run,
        )
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
