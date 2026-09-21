"""Rendering a row into a prompt template.

Lifted from judge-0's `scripts/c1_judge.py`, which is an Ollama-era judge this repo does
not otherwise carry. The two functions are the whole of what the template layer does:
substitute `{column}` placeholders, and refuse a template naming a column the row lacks.
"""

from __future__ import annotations

import json
import re

PLACEHOLDER = re.compile(r"\{([^{}\s]+)\}")
INTEGRAL_FLOAT = re.compile(r"^-?\d+\.0+$")


def render_value(v) -> str:
    """Any JSON scalar, not only a string: an items file may hold `{"age": 70}`, and a
    number that reaches the model as a crash instead of "70" is the worst of both.

    A list or an object is written as JSON. `str()` would hand the model Python -- single
    quotes around every string -- which is not a shape any prompt describes.
    """
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    v = "" if v is None else str(v).strip()
    if not v:
        return "<missing>"
    return v.split(".")[0] if INTEGRAL_FLOAT.match(v) else v   # "63.0" -> "63", lossless


def render_user(template: str, row: dict) -> str:
    missing = [k for k in PLACEHOLDER.findall(template) if k not in row]
    if missing:
        raise KeyError(f"template columns not in row: {missing}")
    return PLACEHOLDER.sub(lambda m: render_value(row[m.group(1)]), template)
