"""Rendering a row into a prompt template.

Lifted from judge-0's `scripts/c1_judge.py`, which is an Ollama-era judge this repo does
not otherwise carry. The two functions are the whole of what the template layer does:
substitute `{column}` placeholders, and refuse a template naming a column the row lacks.
"""

from __future__ import annotations

import re

PLACEHOLDER = re.compile(r"\{([^{}\s]+)\}")
INTEGRAL_FLOAT = re.compile(r"^-?\d+\.0+$")


def render_value(v) -> str:
    """Any JSON scalar, not only a string: an items file may hold `{"age": 70}`, and a
    number that reaches the model as a crash instead of "70" is the worst of both."""
    v = "" if v is None else str(v).strip()
    if not v:
        return "<missing>"
    return v.split(".")[0] if INTEGRAL_FLOAT.match(v) else v   # "63.0" -> "63", lossless


def render_user(template: str, row: dict) -> str:
    missing = [k for k in PLACEHOLDER.findall(template) if k not in row]
    if missing:
        raise KeyError(f"template columns not in row: {missing}")
    return PLACEHOLDER.sub(lambda m: render_value(row[m.group(1)]), template)
