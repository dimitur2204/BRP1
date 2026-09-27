# Validation of A Method to Compensate Multicenter Effects Affecting CT Radiomics

- **Authors:** Orlhac F, Frouin F, Nioche C, Ayache N, Buvat I
- **Year / venue:** 2019, *Radiology* 291(1):53–59
- **DOI:** 10.1148/radiol.2019182023 · PMID 30694160
- **How verified:** PubMed record and abstract. The original ComBat method (Johnson WE, Li C, Rabinovic A. *Biostatistics* 2007;8(1):118–127, DOI 10.1093/biostatistics/kxj037) was confirmed on PubMed (PMID 16632515, including volume/pages).

## Key findings
- Applies ComBat (empirical-Bayes location/scale batch correction) to CT radiomics, where each protocol (kernel, slice thickness) is a batch.
- Phantom: 10 texture patterns scanned with different protocols. Before correction every feature differed by protocol (P < 0.05). After correction no protocol effect was detectable, and PCA clustered by texture, not protocol.
- Patients: features with a significant protocol effect fell from 100% (10/10) to 30% (3/10) in cohort 1 and from 98% (87/89) to 15% (13/89) in cohort 2.
- Main caveat: batch correction assumes batch is not confounded with biology or outcome. Biological covariates should be preserved in the model.

## Relevance to our project
- This is the method behind our R3 harmonisation (Yeo-Johnson then ComBat with batch = manufacturer × kernel group), which cut kernel predictability from 0.89 to 0.54.
- **Caution for the thymus × calcium step:** harmonise only our radiomics. The thymic-health score comes from its own model and cohort-wide percentiling and should not be passed through our ComBat. Check instead whether thymic health differs by kernel/manufacturer, and adjust for it as a covariate if it does.
