"""llmjudge — an LLM judge over an OpenAI-compatible endpoint.

    make-items   CSV + a pool spec -> items.jsonl
    judge        items.jsonl -> results.jsonl + summary.json

It holds no rules, no cards and no knowledge tables: those live in `rule-builder`
(authoring) and `judge-0` (the evaluation pipeline). The interface across that line is
files on disk, never an import, in both directions.
"""

from .run import judge, main

__all__ = ["judge", "main", "items", "run", "template"]
