# Sybil: A Validated Deep Learning Model to Predict Future Lung Cancer Risk From a Single Low-Dose Chest Computed Tomography

- **Authors:** Mikhael PG, Wohlwend J, Yala A, Karstens L, Xiang J, Takigami AK, Bourgouin PP, Chan P, Mrah S, Amayri W, Juan YH, Yang CT, Wan YL, Lin G, Sequist LV, Fintelmann FJ, Barzilay R
- **Year / venue:** 2023, *Journal of Clinical Oncology* 41(12):2191–2200
- **DOI:** 10.1200/JCO.22.01345 · PMID 36634294
- **How verified:** PubMed record and abstract.

## Key findings
- A 3D deep-learning model trained on NLST LDCTs that predicts 1–6-year lung-cancer risk from one scan, with no clinical data or annotations.
- 1-year AUC: 0.92 (NLST held-out, n = 6,282), 0.86 (MGH, n = 8,821), 0.94 (Chang Gung, n = 12,280, including never-smokers).
- **6-year C-index: 0.75 (0.72–0.78) on NLST**, 0.81 at MGH, 0.80 at CGMH.
- The model and annotations are public. My interpretation, not stated in the abstract: the drop from 1-year AUC (0.92) to 6-year C (0.75) suggests much of the short-term performance comes from nodules or early cancers already visible, i.e. detection, not long-term risk.

## Relevance to our project
- This is the upper benchmark on the same data. Image-based lung-cancer risk in NLST reaches C ≈ 0.75, against our clinical model at 0.647 and clinical + calcium at 0.658. Our cardiac and thymic markers are **not competitors to Sybil**. Their value is as interpretable, extra-pulmonary host-risk markers, and a natural next test is whether they add to Sybil.
- A lung-cancer endpoint that includes cancers diagnosed in the first year is dominated by nodules already present at baseline. A landmark analysis (excluding the first 1–2 years) is advisable so that host markers are assessed on true future risk.
