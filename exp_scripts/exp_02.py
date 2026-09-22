"""Run the exp_01 pools against EvidenceMD, the clinical API.

    export EVIDENCEMD_API_KEY=...
    python exp_scripts/exp_02.py --dry-run
    python exp_scripts/exp_02.py --limit 25

Pass ``--run-id <timestamp>`` to continue a previous run.

Same inputs as exp_01 -- the same rows judged by a different model is the comparison
this experiment is for. Every request is billed at $0.20 whatever its length, so
``--limit`` is small by default and the estimated cost is printed before anything is
sent. ``--limit`` takes the first N of the shuffled send order, so it is a random
sample of the pool, not its first rows.

EvidenceMD is OpenAI chat-completions with three differences, all handled below: the key
goes in ``x-api-key``, there is no model listing to check, and there is no JSON schema --
hence ``guided="off"``, with the answer's shape carried by the prompt and checked after
the fact. Its ``response_format: {"type": "json_object"}`` is not a substitute: it returns
the vendor's own ``{"answer": "..."}`` envelope rather than the object the prompt asked
for, so it is not sent (see the comment in main).

``--prompt`` picks the prompt. ``c3`` is exp_01's, system message and all, which is the
like-for-like comparison and the default. ``c3-evidencemd`` is the same task merged into
the user message, with the persona passed as ``specialty`` instead -- the shape every
published EvidenceMD sample uses. Nothing in the docs says a system role is refused, so
both are legitimate; c3 is the one that compares.

Rate limit is 60 requests/min per key; the window starts at 8 and halves on any 429.
If the canary comes back as a rejected request, the API refused a field we send: add
``drop=["seed"]`` or ``drop=["stream_options"]`` here and re-run. Nothing is billed for
a request rejected with 4xx.

See notes/20260921_225044-evidencemd-in-llmjudge-plan.md.
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
RESULTS = ROOT / "exp_results/exp_02"
PRICE = {"evidencemd-fast": 0.20, "evidencemd-pro": 0.20, "evidencemd-deep": 0.25}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", help="UTC timestamp printed by an earlier run")
    parser.add_argument("--model", default="evidencemd-pro", choices=sorted(PRICE))
    parser.add_argument("--prompt", default="c3",
                        help="c3 (exp_01's, JSON only) or a reason-first prompt such as "
                             "c3-evidencemd or c3-reasoning")
    parser.add_argument("--limit", type=int, default=10, metavar="N",
                        help="rows per input file (default 25); at $0.20 a request, 5,000 "
                             "rows is $1,000")
    args = parser.parse_args(argv)

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if run_id in (".", "..") or Path(run_id).name != run_id:
        parser.error("--run-id must be a directory name, not a path")

    api_key = os.environ.get("EVIDENCEMD_API_KEY")
    if not api_key and not args.dry_run:
        parser.error("set EVIDENCEMD_API_KEY first")

    # *_pool.csv is the pool manifest beside a table -- order, arm, index, weight -- and
    # carries none of the columns the prompt names, so judging it is refused. Skipping it
    # here is what keeps that refusal from landing after the first table has been billed.
    items = [p for p in sorted(INPUTS.glob("*.csv")) if not p.stem.endswith("_pool")]
    if not items:
        parser.error(f"no CSV files found in {INPUTS}")

    # No response_format. EvidenceMD's json_object mode does not return YOUR object: it
    # returns its own envelope with the whole answer as a string inside it --
    #     {"answer": "{\"short_reason\": \"no conflict\", \"verdict\": \"consistent\"}"}
    # -- observed in exp_results/exp_02/20260921T173040Z. The judge scans the reply for a
    # JSON object carrying `verdict`; the outer object has only `answer`, and the inner
    # one is escaped inside a string, so every row parse-fails at $0.20 a go. Without the
    # field the answer comes back as itself.
    extra = {"specialty": "Clinical data auditor"}

    # +2 covers this endpoint's canary and any retry; the cap is what stands between a
    # typo and a four-figure bill, so it is always passed.
    budget = args.limit + 2
    out = RESULTS / run_id
    print(f"experiment output: {out}")
    print(f"budget: {len(items)} x {budget} requests x ${PRICE[args.model]:.2f} = "
          f"${len(items) * budget * PRICE[args.model]:.2f} at most")
    for path in items:
        print(f"input: {path.name}")
        code = judge(
            items=str(path),
            out=str(out / path.stem),
            run_tag=f"exp_02-{run_id}-{path.stem}",
            prompt=args.prompt,
            base_url="https://evidencemd.ai/api/v1",
            api_key=api_key,
            auth_header="x-api-key",          # sent raw, no Bearer
            skip_model_check=True,            # EvidenceMD lists no models
            model=args.model,
            guided="off",                     # no json_schema in the API; the prompt carries the shape
            extra=extra,
            stream=True,                      # vendor: non-streaming 504s on long questions
            max_tokens=2048,
            timeout=300,
            limit=args.limit,
            max_requests=budget,
            dry_run=args.dry_run,
        )
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
