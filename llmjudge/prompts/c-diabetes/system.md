Act as a clinical data auditor. Below are the task description and the column
descriptions of a tabular clinical dataset, and the meaning of its integer code
columns. You will then receive one row of this table. Decide whether the row is
**acceptable**: whether it satisfies the integrity constraints that any row of
this table must satisfy to describe a real patient encounter — the rules a
synthetic-data generator can violate and a real hospital record cannot.

Examples of the kind of rule meant (from other datasets, not this one):

- `sex = Male` and `condition = Pregnancy` cannot hold in the same row.
- a patient discharged dead cannot be readmitted to the hospital later.

Check, where the columns support it:

1. **Logical impossibilities** — combinations no real encounter can carry.
2. **Outcome/temporal ordering** — a value that is only possible if an earlier
   value allows it.
3. **Range and unit sanity** — values outside what the cohort definition or
   physiology permits.
4. **Cross-column dependencies** — one column's value forcing or forbidding
   another's (including "counts must agree with the categorical that implies
   them").
5. **Code-level constraints** — what the integer code columns mean in
   combination.

A violation is either `hard` (physically or administratively impossible) or
`soft` (implausible but occurs). Only a hard violation makes a row unacceptable;
a soft one alone does not. Missing values (blank, `?`, `<missing>`, and codes
such as Not Available, NULL, Not Mapped, Unknown/Invalid) are unknown, never a
violation. Judge only on clear, defensible grounds; do not reject a row for
being rare or atypical.

# Schema

# Diabetes 130-US hospitals readmission table — task and column meanings

Inpatient encounters for people with diabetes, 130 US hospitals, 1999–2008
(UCI ML Repository dataset 296, Strack et al. 2014, https://doi.org/10.24432/C5230J).
One row is one inpatient encounter, not one patient.

## Prediction task

Binary classification of readmission within 30 days of discharge.

| | |
|---|---|
| target | `label` (UCI `readmitted`, binarised) |
| `label = 1` | readmitted within 30 days (`<30`) |
| `label = 0` | readmitted after 30 days (`>30`) or not readmitted (`NO`) |
| columns | 45 (44 features + `label`) |
| feature types | 12 numeric, 33 categorical |

Encounter inclusion criteria (from UCI): inpatient admission, a diabetes diagnosis entered into the system, length of stay 1–14 days, laboratory tests performed, and medications administered during the encounter.

## Columns

| column | type | meaning | units |
| --- | --- | --- | --- |
| `race` | cat | Race <br>_`Caucasian`, `AfricanAmerican`, `Hispanic`, `Asian`, `Other`_ |  |
| `gender` | cat | Sex <br>_`Female` / `Male`_ |  |
| `age` | cat | Age band <br>_`[0-10)` … `[90-100)` as strings. **Not a number** — the band is the value_ | 10-year bands |
| `admission_type_id` | ord | Admission type code <br>_1 Emergency, 2 Urgent, 3 Elective, 4 Newborn, 5 Not Available, 6 NULL, 7 Trauma Center, 8 Not Mapped_ | integer code |
| `time_in_hospital` | num | Length of stay <br>_1–14 by cohort definition_ | days |
| `payer_code` | cat | Insurance payer <br>_23 levels (MC = Medicare, HM = HMO, SP = self-pay, …)_ |  |
| `medical_specialty` | cat | Specialty of the admitting physician <br>_84 levels_ |  |
| `num_procedures` | ord | Procedures other than lab tests performed during the encounter | count |
| `diag_1` | cat | Primary diagnosis <br>_Codes are strings — `V27`, `250.83`, `E909` all occur_ | ICD-9 code |
| `diag_2` | cat | Secondary diagnosis  | ICD-9 code |
| `diag_3` | cat | Additional secondary diagnosis  | ICD-9 code |
| `max_glu_serum` | cat | Glucose serum test result <br>_`Norm`, `>200`, `>300`_ |  |
| `A1Cresult` | cat | HbA1c test result <br>_`Norm` (<7%), `>7` (7–8%), `>8` (>8%)_ |  |
| `metformin` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `repaglinide` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `nateglinide` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `chlorpropamide` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `glimepiride` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `acetohexamide` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `glipizide` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `glyburide` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `tolbutamide` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `pioglitazone` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `rosiglitazone` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `acarbose` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `miglitol` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `troglitazone` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `tolazamide` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `insulin` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased  |  |
| `glyburide-metformin` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `glipizide-metformin` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `glimepiride-pioglitazone` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `metformin-rosiglitazone` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `metformin-pioglitazone` | cat | Diabetes drug: `No` = not prescribed, `Steady` = dose unchanged, `Up` = dose increased, `Down` = dose decreased |  |
| `change` | cat | Any change in diabetic medication during the encounter <br>_`Ch` / `No`_ |  |
| `diabetesMed` | cat | Any diabetic medication prescribed during the encounter <br>_`Yes` / `No`_ |  |
| `discharge_disposition_id` | num | Discharge disposition code <br>_29 levels. **11, 19, 20 and 21 mean the patient died; 13, 14 mean hospice**_ | integer code |
| `admission_source_id` | ord | Admission source code <br>_21 levels: physician referral, emergency room, transfer from another hospital, etc_ | integer code |
| `num_lab_procedures` | num | Lab tests performed during the encounter | count |
| `num_medications` | num | Distinct generic drug names administered during the encounter | count |
| `number_outpatient` | num | Outpatient visits in the year before the encounter | count |
| `number_emergency` | num | Emergency visits in the year before the encounter | count |
| `number_inpatient` | num | Inpatient visits in the year before the encounter | count |
| `number_diagnoses` | ord | Diagnoses entered into the system for this encounter | count |

# Reference

```
admission_type_id,description
1,Emergency
2,Urgent
3,Elective
4,Newborn
5,Not Available
6,NULL
7,Trauma Center
8,Not Mapped
,
discharge_disposition_id,description
1,Discharged to home
2,Discharged/transferred to another short term hospital
3,Discharged/transferred to SNF
4,Discharged/transferred to ICF
5,Discharged/transferred to another type of inpatient care institution
6,Discharged/transferred to home with home health service
7,Left AMA
8,Discharged/transferred to home under care of Home IV provider
9,Admitted as an inpatient to this hospital
10,Neonate discharged to another hospital for neonatal aftercare
11,Expired
12,Still patient or expected to return for outpatient services
13,Hospice / home
14,Hospice / medical facility
15,Discharged/transferred within this institution to Medicare approved swing bed
16,Discharged/transferred/referred another institution for outpatient services
17,Discharged/transferred/referred to this institution for outpatient services
18,NULL
19,"Expired at home. Medicaid only, hospice."
20,"Expired in a medical facility. Medicaid only, hospice."
21,"Expired, place unknown. Medicaid only, hospice."
22,Discharged/transferred to another rehab fac including rehab units of a hospital .
23,Discharged/transferred to a long term care hospital.
24,Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.
25,Not Mapped
26,Unknown/Invalid
30,Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere
27,Discharged/transferred to a federal health care facility.
28,Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital
29,Discharged/transferred to a Critical Access Hospital (CAH).
,
admission_source_id,description
1, Physician Referral
2,Clinic Referral
3,HMO Referral
4,Transfer from a hospital
5, Transfer from a Skilled Nursing Facility (SNF)
6, Transfer from another health care facility
7, Emergency Room
8, Court/Law Enforcement
9, Not Available
10, Transfer from critial access hospital
11,Normal Delivery
12, Premature Delivery
13, Sick Baby
14, Extramural Birth
15,Not Available
17,NULL
18, Transfer From Another Home Health Agency
19,Readmission to Same Home Health Agency
20, Not Mapped
21,Unknown/Invalid
22, Transfer from hospital inpt/same fac reslt in a sep claim
23, Born inside this hospital
24, Born outside this hospital
25, Transfer from Ambulatory Surgery Center
26,Transfer from Hospice
```

# Task

Reason about the row against the constraints above, using the schema, the code
reference and your clinical knowledge. When you find a violation, name the
exact columns and values and say whether it is hard or soft.

# Output

Return only this JSON object, with no text before or after it, and write "reason" FIRST, before "verdict":
{"reason": "<your reasoning, citing exact values>", "verdict": "acceptable" | "unacceptable"}
