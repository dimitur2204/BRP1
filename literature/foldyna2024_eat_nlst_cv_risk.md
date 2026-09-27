# Deep learning analysis of epicardial adipose tissue to predict cardiovascular risk in heavy smokers

- **Authors:** Foldyna B, Hadzic I, Zeleznik R, Langenbach MC, Raghu VK, Mayrhofer T, Lu MT, Aerts HJWL
- **Year / venue:** 2024, *Communications Medicine* 4(1):44
- **DOI:** 10.1038/s43856-024-00475-1 · PMID 38480863 · PMCID PMC10937640
- **How verified:** PubMed abstract and Europe PMC full text.

## Key findings
- NLST CT arm, n = 24,090 (59% men, 61 ± 5 years), follow-up 12.3 years. Deep-learning EAT segmentation on baseline non-gated LDCT. Mean BSA-indexed EAT volume 70.3 ± 24.6 cm³/m², density −77.7 ± 5.2 HU.
- EAT volume and density were independently associated with all-cause mortality (aHR 1.10 and 1.38) and CV mortality (aHR 1.14 and 1.78). Adjustment covered demographics, **smoking status (current vs former), pack-years**, CV history, diabetes, hypertension, education, BMI and **CAC**.
- Per 10 cm³/m² EAT volume: all-cause HR 1.19 (1.17–1.21), CV HR 1.27 (1.23–1.30) before full adjustment.
- **Correlates of EAT in NLST:** higher with age, male sex, BMI and higher CAC, **higher in former than current smokers, and rising with pack-years** (all P ≤ 0.007). Lung-cancer incidence was not analysed.

## Relevance to our project
- EAT in NLST is **confounded by smoking in two directions**: current smokers have *less* EAT (lower weight), while pack-years relate to *more* EAT. With only current/former status, our pericardial-fat feature partly encodes smoking status and BMI. It must be adjusted for BMI, which we should obtain from CDAS/IDC.
- EAT is prognostic beyond CAC for CV death, but nobody has tested it for lung-cancer incidence. For our purposes it is a vascular/metabolic marker different from calcium.
