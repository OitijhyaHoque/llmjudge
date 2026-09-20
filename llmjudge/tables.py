"""CSV reading and writing. Rows are dicts of str; blanks stay blank.

The tables in `data/tabular/` top out at 253,680 rows, so the stdlib `csv`
module is the right tool and a dataframe library would only add a dependency.
"""

from __future__ import annotations

import csv
import sys
from typing import Iterator

Row = dict[str, str]

# The widest column in the corpus is diabetes130's `medical_specialty`; raising the
# field limit costs nothing and avoids a surprise on a table we have not seen.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def read_rows(path: str) -> list[Row]:
    """Read a CSV into a list of dicts, preserving the file's own strings."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def iter_rows(path: str) -> Iterator[Row]:
    """Stream a CSV. Use for the 100k+ row tables when you only need one pass."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)


def header(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f))


def write_rows(path: str, rows: list[Row], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise ValueError("nothing to write and no fieldnames given")
    fieldnames = fieldnames or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def as_float(value: str | None) -> float | None:
    """Parse a cell as a float, or None if it is blank or not numeric.

    Every check treats None as "does not apply to this row" rather than as a
    failure, so that missingness is never silently scored as a defect.
    """
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None
