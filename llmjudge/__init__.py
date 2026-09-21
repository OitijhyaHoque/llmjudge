"""llmjudge — an LLM judge over an OpenAI-compatible endpoint.

    make-items   CSV + a pool spec -> items.jsonl
    judge        items.jsonl -> results.jsonl + summary.json

It holds no rules, no cards and no knowledge tables. Whatever authors the rules and
whatever reads the verdicts back stay outside: the interface is files on disk, never an
import, in both directions.
"""

from .run import judge, main

__all__ = ["judge", "main"]
