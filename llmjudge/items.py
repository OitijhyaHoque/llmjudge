#!/usr/bin/env python3
"""Build an items JSONL from CSVs and a pool spec.

    llmjudge make-items --spec configs/pool.diabetes130.toml --pool full --out items.jsonl

Choosing which rows to judge is sampling, not judging, so it could have lived somewhere
else. It lives here because the only thing it needs is CSV columns -- no card, no checks,
no knowledge tables -- and a separate home for it made a dead repository load-bearing.

A spec names its tables and its groups. A table is a CSV of rows to judge, optionally
paired with a second CSV of per-row attributes (a rules `decisions.csv`, a labelling, a
clustering) that selection and stratification read but the model never sees:

    [tables.ctgan_split]
    rows  = "runs/x/normalize/ctgan_split/table.csv"
    attrs = "runs/x/rules/ctgan_split/decisions.csv"      # optional

    [groups.kept]
    table = "ctgan_split"
    alloc = "proportional"                  # or "equal"
    select = { kept = "1" }                 # column = value, or column = [v1, v2]
    stratum = "{decision}/label={label}"    # any column of attrs or rows

    [sizes.full]
    kept = 0                                # 0 = every eligible row
    positive = 500

Rows are drawn per stratum with a seeded RNG, so the same spec and seed give the same
file. `weight` is the stratum's sampling weight; `llmjudge` carries it through to
`summary.json`, which is what turns a stratified pool into a population estimate.

Only `fields` reaches the prompt. The arm, the row index and every attrs column stay out
of it, so the model cannot see what a rule decided about the row it is judging.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tomllib
from typing import Iterator

EXIT_OK, EXIT_CONFIG = 0, 2


class ConfigError(Exception):
    pass


def iter_rows(path: str) -> Iterator[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def allocate(sizes: dict[str, int], n: int, alloc: str) -> dict[str, int]:
    """Rows to take per stratum. n <= 0 or n >= total takes everything."""
    total = sum(sizes.values())
    keys = sorted(sizes)
    if n <= 0 or n >= total:
        return dict(sizes)
    if alloc == "equal":
        take, left = dict.fromkeys(keys, 0), n
        while left:
            moved = False
            for k in keys:
                if left and take[k] < sizes[k]:
                    take[k] += 1
                    left -= 1
                    moved = True
            if not moved:
                break
        return take
    exact = {k: n * sizes[k] / total for k in keys}
    take = {k: min(sizes[k], int(exact[k])) for k in keys}
    left = n - sum(take.values())
    for k in sorted(keys, key=lambda k: (-(exact[k] - int(exact[k])), k)):
        if left <= 0:
            break
        if take[k] < sizes[k]:
            take[k] += 1
            left -= 1
    return take


def matches(select: dict, attrs: dict) -> bool:
    for col, want in select.items():
        if col not in attrs:
            raise ConfigError(f"select column {col!r} is not in the table "
                              f"(columns: {', '.join(sorted(attrs)[:8])}...)")
        got = attrs[col]
        hit = got == want if isinstance(want, str) else got in want
        if not hit:
            return False
    return True


def load_spec(path: str, root: str) -> dict:
    with open(path, "rb") as f:
        spec = tomllib.load(f)
    for section in ("tables", "groups", "sizes"):
        if not spec.get(section):
            raise ConfigError(f"{path}: no [{section}] section")
    for name, t in spec["tables"].items():
        if "rows" not in t:
            raise ConfigError(f"{path}: table {name!r} has no rows = ...")
        for key in ("rows", "attrs"):
            if t.get(key) and not os.path.isabs(t[key]):
                t[key] = os.path.normpath(os.path.join(root, t[key]))
    for name, g in spec["groups"].items():
        for key in ("table", "select", "stratum"):
            if key not in g:
                raise ConfigError(f"{path}: group {name!r} has no {key} = ...")
        if g["table"] not in spec["tables"]:
            raise ConfigError(f"{path}: group {name!r} names table {g['table']!r}, "
                              f"which has no [tables.{g['table']}]")
        g.setdefault("alloc", "proportional")
        if g["alloc"] not in ("proportional", "equal"):
            raise ConfigError(f"{path}: group {name!r} alloc must be proportional or equal")
    return spec


def build_items(spec: dict, sizes: dict[str, int], seed: int) -> tuple[list[dict], dict]:
    """-> (items sorted by id, input provenance).

    Two passes per table: the first streams both CSVs together and records only the index
    of every eligible row, the second re-reads and keeps whole rows for the sample. The
    tables run to 100k+ rows and only a few thousand are ever drawn.
    """
    by_table: dict[str, list[str]] = {}
    for g in spec["groups"]:                               # spec order, not sorted
        by_table.setdefault(spec["groups"][g]["table"], []).append(g)

    items: list[dict] = []
    inputs: dict[str, dict] = {}
    for table, groups in by_table.items():
        t = spec["tables"][table]
        rows_path, attrs_path = t["rows"], t.get("attrs")
        for p in (rows_path, attrs_path):
            if p and not os.path.exists(p):
                raise ConfigError(f"missing input {p}")
        inputs[table] = {"rows": rows_path, "rows_sha": file_sha(rows_path)}
        if attrs_path:
            inputs[table].update(attrs=attrs_path, attrs_sha=file_sha(attrs_path))

        strata: dict[str, dict[str, list[int]]] = {g: {} for g in groups}
        stream = (zip(iter_rows(attrs_path), iter_rows(rows_path), strict=True) if attrs_path
                  else ((r, r) for r in iter_rows(rows_path)))
        try:
            for i, (a, r) in enumerate(stream):
                if a.get("index") not in (None, str(i)):
                    raise ConfigError(f"{attrs_path or rows_path}: row {i} has "
                                      f"index {a['index']} -- attrs and rows are misaligned")
                merged = None
                for g in groups:
                    if not matches(spec["groups"][g]["select"], a):
                        continue
                    if merged is None:
                        merged = {**r, **a}
                    try:
                        key = spec["groups"][g]["stratum"].format_map(merged)
                    except KeyError as e:
                        raise ConfigError(f"group {g!r}: stratum names column {e}, which "
                                          f"is in neither {table} nor its attrs") from e
                    strata[g].setdefault(key, []).append(i)
        except ValueError as e:
            raise ConfigError(f"{table}: rows and attrs differ in length ({e})") from e

        wanted: dict[int, list[tuple[str, str, float]]] = {}
        for g in groups:
            if g not in sizes:
                raise ConfigError(f"no size for group {g!r} in this pool")
            st = strata[g]
            take = allocate({k: len(v) for k, v in st.items()}, sizes[g],
                            spec["groups"][g]["alloc"])
            for k in sorted(st):
                idxs = st[k]
                random.Random(f"{seed}|{g}|{k}").shuffle(idxs)
                n = take.get(k, 0)
                weight = round(len(idxs) / n, 6) if n else 0.0
                for i in idxs[:n]:
                    wanted.setdefault(i, []).append((g, k, weight))

        seen: dict[str, str] = {}
        for i, row in enumerate(iter_rows(rows_path)):
            for g, k, weight in wanted.get(i, ()):
                item_id = f"{table}:{i}"
                if item_id in seen:
                    raise ConfigError(f"row {item_id} is in group {seen[item_id]!r} and "
                                      f"{g!r}; llmjudge needs unique ids, so a row cannot "
                                      "be two items")
                seen[item_id] = g
                items.append({"id": item_id, "fields": dict(row),
                              "group": g, "stratum": k, "weight": weight})

    items.sort(key=lambda o: o["id"])
    return items, inputs


def write_items(path: str, items: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def parse_sizes(text: str, groups: list[str]) -> dict[str, int]:
    out = {}
    for part in text.split(","):
        g, _, n = part.partition("=")
        if g.strip() not in groups:
            raise ConfigError(f"--sizes group {g.strip()!r} not in {sorted(groups)}")
        out[g.strip()] = int(n)
    return out


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="llmjudge make-items",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, help="pool spec TOML")
    ap.add_argument("--root", default="",
                    help="prefix for relative paths in the spec, so one spec works from "
                         "a checkout, a Drive mount or a colleague's machine "
                         "(default: the spec's own directory)")
    ap.add_argument("--pool", help="which [sizes.<name>] to use (default: the only one)")
    ap.add_argument("--sizes", default="", help="override: group=N,group=N  (0 = every row)")
    ap.add_argument("--seed", type=int, default=42, help="pool sampling seed")
    ap.add_argument("--out", required=True, help="items JSONL to write")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        spec = load_spec(args.spec, args.root or os.path.dirname(os.path.abspath(args.spec)))
        pools = spec["sizes"]
        if args.pool is None:
            if len(pools) != 1:
                raise ConfigError(f"--pool is one of {sorted(pools)}")
            args.pool = next(iter(pools))
        if args.pool not in pools:
            raise ConfigError(f"--pool {args.pool!r} is not one of {sorted(pools)}")
        sizes = dict(pools[args.pool])
        if args.sizes:
            sizes.update(parse_sizes(args.sizes, list(spec["groups"])))
        items, inputs = build_items(spec, sizes, args.seed)
    except (ConfigError, OSError, tomllib.TOMLDecodeError) as e:
        print(f"make-items: refused: {e}", file=sys.stderr, flush=True)
        return EXIT_CONFIG

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_items(args.out, items)
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"spec": args.spec, "spec_sha": file_sha(args.spec), "pool": args.pool,
                   "sizes": sizes, "seed": args.seed, "items": len(items),
                   "inputs": inputs}, f, indent=2)
        f.write("\n")

    counts: dict[str, int] = {}
    for obj in items:
        counts[obj["group"]] = counts.get(obj["group"], 0) + 1
    print(f"{args.out}: {len(items)} items  "
          + "  ".join(f"{g}={counts[g]}" for g in sorted(counts)))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
