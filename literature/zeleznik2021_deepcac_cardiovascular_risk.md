# Deep convolutional neural networks to predict cardiovascular risk from computed tomography

- **Authors:** Zeleznik R, Foldyna B, Eslami P, Weiss J, Alexander I, Taron J, Parmar C, Alvi RM, Banerji D, Uno M, Kikuchi Y, Karady J, Zhang L, Scholtz JE, Mayrhofer T, Lyass A, Mahoney TF, Massaro JM, Vasan RS, Douglas PS, Hoffmann U, Lu MT, Aerts HJWL
- **Year / venue:** 2021, *Nature Communications* 12:715
- **DOI:** 10.1038/s41467-021-20966-2 · PMID 33514711 · PMCID PMC7846726
- **How verified:** PubMed abstract and full text (Europe PMC XML, results table).

## Key findings
- The DeepCAC system (a 3D U-Net) automatically quantifies coronary calcium on gated and non-gated CT. It was tested in 20,084 people from FHS, NLST, PROMISE and ROMICAT-II.
- Against manual scoring in 5,521 test patients (including NLST n = 396): Spearman 0.92, with good risk-group agreement and robust test–retest.
- **NLST** (n = 14,959, 288 ASCVD deaths): high vs very-low DL-CAC group HR 5.98 (3.86–9.26) unadjusted, 4.34 (2.75–6.84) age/sex-adjusted, and **3.87 (2.45–6.11)** adjusted for age, sex, diabetes, hypertension, prior heart disease and stroke. Smoking/pack-years were not in that model.

## Relevance to our project
- This is the standard automated CAC on NLST from the thymus-paper group. The expert annotations of 390 NLST scans it produced are also the DeepCAC2 test set. It is the natural external benchmark for our calcium proxy.
- Its adjustment set leaves out smoking intensity. For CV outcomes that residual confounding was accepted, but for our lung-cancer endpoint smoking is the main confounder.
