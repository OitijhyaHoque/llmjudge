Act as a clinical data auditor. Below are the task description and the column
descriptions of a tabular clinical dataset. You will then receive one row of
this table. Decide whether the row is **acceptable**: whether it satisfies the
integrity constraints that any row of this table must satisfy to describe a
real patient — the rules a synthetic-data generator can violate and a real
hospital record cannot.

Examples of the kind of rule meant (from other datasets, not this one):

- `sex = Male` and `condition = Pregnancy` cannot hold in the same row.
- a patient discharged dead cannot be readmitted to the hospital later.

Check, where the columns support it:

1. **Logical impossibilities** — combinations no real patient can carry.
2. **Outcome/temporal ordering** — a value that is only possible if an earlier
   value allows it.
3. **Range and unit sanity** — values outside what the cohort definition or
   physiology permits.
4. **Cross-column dependencies** — one column's value forcing or forbidding
   another's (including "counts must agree with the categorical that implies
   them").
5. **Code-level constraints** — what the coded columns mean in combination.

A violation is either `hard` (physically or administratively impossible) or
`soft` (implausible but occurs). Only a hard violation makes a row unacceptable;
a soft one alone does not. Missing values (blank, `?`, `<missing>`) are unknown,
never a violation. Judge only on clear, defensible grounds; do not reject a row
for being rare or atypical.

# Schema

# Indian Liver Patient Dataset (ILPD) — task and column meanings

Patients from north-east Andhra Pradesh, India, with age, sex and a liver-function blood panel
(UCI ML Repository dataset 225, Ramana & Venkateswarlu, https://doi.org/10.24432/C5D02C).
One row is one patient. The class was assigned by clinical experts.

## Prediction task

Binary classification of liver disease.

| | |
|---|---|
| target | `label` (UCI `Selector`) |
| `label = 0` | **liver patient** (UCI `Selector = 1`) |
| `label = 1` | **not a liver patient** (UCI `Selector = 2`) |
| columns | 11 (10 features + `label`) |
| feature types | 9 numeric, 1 categorical (`Gender`) |

Age is top-coded by UCI: any patient older than 89 is recorded as `90`. The cohort includes children.
UCI does not state units for the lab columns; the units below are the standard reporting units for these tests.

## Columns

| column | type | meaning | units |
| --- | --- | --- | --- |
| `Gender` | cat | Sex <br>_`Male` / `Female`_ |  |
| `Age` | num | Age <br>_`90` means 90 or older_ | years |
| `TB` | num | Total bilirubin | mg/dL |
| `DB` | num | Direct (conjugated) bilirubin | mg/dL |
| `Alkphos` | num | Alkaline phosphatase (ALP) | IU/L |
| `Sgpt` | num | Alanine aminotransferase (ALT / SGPT) | IU/L |
| `Sgot` | num | Aspartate aminotransferase (AST / SGOT) | IU/L |
| `TP` | num | Total serum protein | g/dL |
| `ALB` | num | Serum albumin | g/dL |
| `A/G Ratio` | num | Albumin-to-globulin ratio | ratio |

# Task

Reason about the row against the constraints above, using the schema and your
clinical knowledge. When you find a violation, name the exact columns and
values and say whether it is hard or soft.

# Output

Return only this JSON object, with no text before or after it, and write "reason" FIRST, before "verdict":
{"reason": "<your reasoning, citing exact values>", "verdict": "acceptable" | "unacceptable"}
