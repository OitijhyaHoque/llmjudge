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

# Cervical cancer (risk factors) table — task and column meanings

Patients of the Hospital Universitario de Caracas, Venezuela: demographic information, habits and historic medical records
(UCI ML Repository dataset 383, Fernandes, Cardoso & Fernandes 2017, https://doi.org/10.24432/C5Z310).
One row is one patient. Values come from a questionnaire; several patients declined to answer some questions for privacy reasons.

## Prediction task

Binary classification of a positive cervical biopsy.

| | |
|---|---|
| target | `label` (UCI `Biopsy`) |
| `label = 1` | positive biopsy |
| `label = 0` | negative biopsy |
| columns | 29 (28 features + `label`) |
| feature types | 28 numeric: 18 are 0/1 flags, 4 are integer counts, 6 are ages or durations |

Not in this table: the other three UCI targets (`Hinselmann`, `Schiller`, `Citology`), the STD flags `STDs:cervical condylomatosis` and `STDs:AIDS`, and `STDs: Time since first diagnosis` / `STDs: Time since last diagnosis`.

## Columns

| column | type | meaning | units |
| --- | --- | --- | --- |
| `Number of sexual partners` | ord | Lifetime number of sexual partners | count |
| `Num of pregnancies` | ord | Number of pregnancies | count |
| `Smokes` | bin | Smoker <br>_`1` = yes, `0` = no_ | 0/1 |
| `Hormonal Contraceptives` | bin | Uses or used hormonal contraceptives <br>_`1` = yes, `0` = no_ | 0/1 |
| `IUD` | bin | Uses or used an intrauterine device <br>_`1` = yes, `0` = no_ | 0/1 |
| `STDs` | bin | Has had a sexually transmitted disease <br>_`1` = yes, `0` = no_ | 0/1 |
| `STDs (number)` | ord | Number of sexually transmitted diseases | count |
| `STDs:condylomatosis` | bin | History of condylomatosis (genital warts) | 0/1 |
| `STDs:vaginal condylomatosis` | bin | History of vaginal condylomatosis | 0/1 |
| `STDs:vulvo-perineal condylomatosis` | bin | History of vulvo-perineal condylomatosis | 0/1 |
| `STDs:syphilis` | bin | History of syphilis | 0/1 |
| `STDs:pelvic inflammatory disease` | bin | History of pelvic inflammatory disease | 0/1 |
| `STDs:genital herpes` | bin | History of genital herpes | 0/1 |
| `STDs:molluscum contagiosum` | bin | History of molluscum contagiosum | 0/1 |
| `STDs:HIV` | bin | History of HIV infection | 0/1 |
| `STDs:Hepatitis B` | bin | History of hepatitis B | 0/1 |
| `STDs:HPV` | bin | History of human papillomavirus infection | 0/1 |
| `STDs: Number of diagnosis` | ord | Number of STD diagnoses <br>_UCI gives no further definition_ | count |
| `Dx:Cancer` | bin | Previous diagnosis of cancer <br>_UCI gives no further definition_ | 0/1 |
| `Dx:CIN` | bin | Previous diagnosis of cervical intraepithelial neoplasia <br>_UCI gives no further definition_ | 0/1 |
| `Dx:HPV` | bin | Previous diagnosis of HPV <br>_UCI gives no further definition_ | 0/1 |
| `Dx` | bin | Any previous diagnosis <br>_UCI gives no further definition_ | 0/1 |
| `Age` | num | Age | years |
| `First sexual intercourse` | num | Age at first sexual intercourse | years |
| `Smokes (years)` | num | Duration of smoking | years |
| `Smokes (packs/year)` | num | Amount smoked <br>_Named "packs/year" in the source; UCI gives no further definition_ | packs/year |
| `Hormonal Contraceptives (years)` | num | Duration of hormonal contraceptive use | years |
| `IUD (years)` | num | Duration of IUD use | years |

# Task

Reason about the row against the constraints above, using the schema and your
clinical knowledge. When you find a violation, name the exact columns and
values and say whether it is hard or soft.

# Output

Return only this JSON object, with no text before or after it, and write "reason" FIRST, before "verdict":
{"reason": "<your reasoning, citing exact values>", "verdict": "acceptable" | "unacceptable"}
