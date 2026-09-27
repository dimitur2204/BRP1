# Future data request: full NLST clinical data via CDAS

**Status:** not requested yet. To propose to the mentor (noted 2026-09-26).

## Why

The public TCIA IDC-780 package we use (`data/external/nlst_780/`) only has
age, sex, race, smoking *status* (current/former), screening results,
lung-cancer outcomes and CT abnormality reads. The full NLST dataset has
variables that would materially strengthen the heart-radiomics project:

| Need | Why it matters for us |
|---|---|
| **Pack-years** / smoking duration and intensity | The biggest residual confounder. Without it, a heart→lung-cancer signal may just be proxying cumulative smoking exposure (coronary calcium tracks it). |
| **Death, cause of death, follow-up time** | Enables all-cause and cardiovascular mortality outcomes, where heart radiomics (calcium, pericardial fat) have strong prior evidence. This is a natural positive-control outcome, and death is a competing risk for lung-cancer incidence. |
| **Medical history** (COPD/emphysema, heart disease/MI, hypertension, diabetes, stroke) | Covariates, and emphysema tests the hyperinflation → heart-shape route. |
| **BMI** (height/weight) | Confounder for pericardial fat and heart size. |

Variable names to look for (from memory of the NLST dictionaries;
**verify against the CDAS data dictionary** before relying on them):
`pkyr`, `smokeyr`, `smokeday`, `age_quit`, `deathstat`, `death_days`,
`fup_days`, `finaldeathlc`, `dcficd`, `diagcopd`, `diagemph`, `diaghear`,
`diaghype`, `diagdiab`, `diagstro`, `height`, `weight`.

## How

- Portal: NCI Cancer Data Access System (CDAS), https://cdas.cancer.gov/nlst/
- Free. Requires a short project proposal and a signed data transfer
  agreement. Approval typically takes weeks, so apply early.
- The mentor (or the group's PI) is the natural applicant. Check first
  whether an existing approved CDAS project for the group already covers
  these variables.

## Plan impact once available

- Add pack-years + comorbidities to the clinical baseline and adjusted models
  (R3 univariate, R4 clinical Cox, R5 ΔC).
- Add mortality outcomes as a secondary analysis (heart radiomics → all-cause /
  CVD death), and a competing-risks check for lung-cancer incidence.
