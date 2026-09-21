"""make-items: selection, stratification, weights, and the refusals.

These run on small synthetic CSVs. The check that make-items reproduces the pools drawn
by the earlier standalone script needs a 2.9 GB tree of real runs and so cannot live
here; it was run separately on 2026-09-21 over three pools -- 100, 55,039 and 5,500 rows
-- and all three came out byte for byte identical.
"""

import csv
import json
import os
import tempfile
import unittest

from llmjudge import items


def write_csv(path, rows, fieldnames=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.rows_path = os.path.join(self.dir, "table.csv")
        self.attrs_path = os.path.join(self.dir, "decisions.csv")
        # 20 rows: labels alternate, the first 12 are kept, 4 are rejected by a named check
        write_csv(self.rows_path,
                  [{"age": 40 + i, "label": i % 2, "note": "x"} for i in range(20)])
        write_csv(self.attrs_path,
                  [{"index": i,
                    "kept": "1" if i < 12 else "0",
                    "decision": "accept" if i < 12 else "reject",
                    "checks": "kb.sex" if 12 <= i < 16 else ("kb.age" if i >= 16 else "")}
                   for i in range(20)])

    def spec(self, body):
        path = os.path.join(self.dir, "pool.toml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(body.replace("ROWS", self.rows_path).replace("ATTRS", self.attrs_path))
        return path

    def build(self, body, pool=None, sizes="", seed=42):
        out = os.path.join(self.dir, "items.jsonl")
        argv = ["--spec", self.spec(body), "--out", out, "--seed", str(seed)]
        if pool:
            argv += ["--pool", pool]
        if sizes:
            argv += ["--sizes", sizes]
        code = items.main(argv)
        if code != items.EXIT_OK:
            return code, []
        with open(out, encoding="utf-8") as f:
            return code, [json.loads(line) for line in f]


TWO_GROUPS = """
[tables.t]
rows = "ROWS"
attrs = "ATTRS"

[groups.kept]
table = "t"
alloc = "proportional"
select = { kept = "1" }
stratum = "label={label}"

[groups.positive]
table = "t"
alloc = "equal"
select = { decision = "reject", checks = ["kb.sex", "kb.age"] }
stratum = "{checks}"

[sizes.pilot]
kept = 6
positive = 4
"""


class TestBuild(Base):
    def test_groups_strata_and_weights(self):
        code, got = self.build(TWO_GROUPS)
        self.assertEqual(code, items.EXIT_OK)
        self.assertEqual(len(got), 10)

        kept = [o for o in got if o["group"] == "kept"]
        self.assertEqual(len(kept), 6)
        # 12 eligible rows, 6 per label, proportional -> 3 and 3, weight 6/3
        self.assertEqual(sorted(o["stratum"] for o in kept),
                         ["label=0"] * 3 + ["label=1"] * 3)
        self.assertEqual({o["weight"] for o in kept}, {2.0})

        pos = [o for o in got if o["group"] == "positive"]
        self.assertEqual(sorted(o["stratum"] for o in pos),
                         ["kb.age"] * 2 + ["kb.sex"] * 2)
        self.assertEqual({o["weight"] for o in pos}, {2.0})

    def test_only_fields_carries_the_row(self):
        _, got = self.build(TWO_GROUPS)
        o = got[0]
        self.assertEqual(sorted(o), ["fields", "group", "id", "stratum", "weight"])
        self.assertEqual(sorted(o["fields"]), ["age", "label", "note"])
        # nothing a rule decided reaches the model
        for key in ("kept", "decision", "checks", "index"):
            self.assertNotIn(key, o["fields"])

    def test_ids_are_table_and_index_and_sorted(self):
        _, got = self.build(TWO_GROUPS)
        self.assertEqual([o["id"] for o in got], sorted(o["id"] for o in got))
        for o in got:
            table, _, i = o["id"].partition(":")
            self.assertEqual(table, "t")
            self.assertEqual(got and o["fields"]["age"], str(40 + int(i)))

    def test_same_seed_same_file_different_seed_different_rows(self):
        _, a = self.build(TWO_GROUPS)
        _, b = self.build(TWO_GROUPS)
        _, c = self.build(TWO_GROUPS, seed=7)
        self.assertEqual(a, b)
        self.assertNotEqual([o["id"] for o in a], [o["id"] for o in c])

    def test_size_zero_takes_every_eligible_row(self):
        _, got = self.build(TWO_GROUPS, sizes="kept=0")
        self.assertEqual(len([o for o in got if o["group"] == "kept"]), 12)

    def test_no_attrs_file_selects_on_the_rows_themselves(self):
        code, got = self.build("""
[tables.t]
rows = "ROWS"

[groups.ones]
table = "t"
select = { label = "1" }
stratum = "all"

[sizes.pilot]
ones = 4
""")
        self.assertEqual(code, items.EXIT_OK)
        self.assertEqual(len(got), 4)
        self.assertTrue(all(o["fields"]["label"] == "1" for o in got))

    def test_meta_records_the_inputs(self):
        self.build(TWO_GROUPS)
        with open(os.path.join(self.dir, "items.jsonl.meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["items"], 10)
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(len(meta["spec_sha"]), 64)
        self.assertEqual(len(meta["inputs"]["t"]["rows_sha"]), 64)


class TestRefusals(Base):
    def refused(self, body, **kw):
        code, _ = self.build(body, **kw)
        self.assertEqual(code, items.EXIT_CONFIG)

    def test_a_row_in_two_groups_is_refused(self):
        self.refused(TWO_GROUPS.replace(
            'select = { decision = "reject", checks = ["kb.sex", "kb.age"] }',
            'select = { kept = "1" }'))

    def test_misaligned_attrs_are_refused(self):
        write_csv(self.attrs_path, [{"index": i + 3, "kept": "1", "decision": "accept",
                                     "checks": ""} for i in range(20)])
        self.refused(TWO_GROUPS)

    def test_different_lengths_are_refused(self):
        write_csv(self.attrs_path, [{"index": i, "kept": "1", "decision": "accept",
                                     "checks": ""} for i in range(5)])
        self.refused(TWO_GROUPS)

    def test_unknown_select_column_is_refused(self):
        self.refused(TWO_GROUPS.replace('select = { kept = "1" }',
                                        'select = { nosuch = "1" }'))

    def test_unknown_stratum_column_is_refused(self):
        self.refused(TWO_GROUPS.replace('stratum = "label={label}"',
                                        'stratum = "{nosuch}"'))

    def test_group_naming_a_missing_table_is_refused(self):
        self.refused(TWO_GROUPS.replace('[groups.kept]\ntable = "t"',
                                        '[groups.kept]\ntable = "nosuch"'))

    def test_missing_size_for_a_group_is_refused(self):
        self.refused(TWO_GROUPS.replace("positive = 4\n", ""))

    def test_ambiguous_pool_is_refused(self):
        self.refused(TWO_GROUPS + "\n[sizes.full]\nkept = 12\npositive = 4\n")

    def test_missing_input_file_is_refused(self):
        os.remove(self.rows_path)
        self.refused(TWO_GROUPS)


if __name__ == "__main__":
    unittest.main()
