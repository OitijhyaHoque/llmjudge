"""Render a row into a prompt template.

Two functions, and they are the whole template layer: substitute the `{column}`
placeholders, and refuse a template that names a column the row does not have.

    render_user("Patient {age}, diagnosis {diag_1}",
                {"age": "[70-80)", "diag_1": "250.83"})
    -> "Patient [70-80), diagnosis 250.83"
"""

from __future__ import annotations

import json
import re

# A column name may hold spaces ("Smokes (years)"), but not a newline or a leading or
# trailing space, so `{ ... }` in prose is still not a placeholder.
PLACEHOLDER = re.compile(r"\{([^{}\s](?:[^{}\n]*[^{}\s])?)\}")
INTEGRAL_FLOAT = re.compile(r"^-?\d+\.0+$")


def render_value(v) -> str:
    """One field value, as the prompt should see it.

    A field may be any JSON value, not only a string: an items file may hold
    `{"age": 70}`. Lists and objects are written as JSON, because `str()` would hand the
    model Python -- single quotes around every string -- which no prompt describes.

        70         -> "70"
        63.0       -> "63"
        None, ""   -> "<missing>"
        ["a", 1]   -> '["a", 1]'
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
