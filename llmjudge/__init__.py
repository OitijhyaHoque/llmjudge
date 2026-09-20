"""llmjudge — an LLM judge over an OpenAI-compatible endpoint.

This repository judges rows. It holds no rules, no cards, no knowledge tables and no
dataset-specific column names: those live in `rule-builder` (authoring) and `judge-0`
(execution and the evaluation pipeline). The interface across that line is files.
"""

__all__ = ["run", "prompts", "tables"]
