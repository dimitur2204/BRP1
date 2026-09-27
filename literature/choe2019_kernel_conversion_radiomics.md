# Deep Learning-based Image Conversion of CT Reconstruction Kernels Improves Radiomics Reproducibility for Pulmonary Nodules or Masses

- **Authors:** Choe J, Lee SM, Do KH, Lee G, Lee JG, Lee SM, Seo JB
- **Year / venue:** 2019, *Radiology* 292(2):365–373
- **DOI:** 10.1148/radiol.2019181960 · PMID 31210613
- **How verified:** PubMed record and abstract.

## Key findings
- 104 patients, each with the same acquisition reconstructed with a soft (B30f) and a sharp (B50f) kernel. 702 radiomic features from lung nodules.
- Same kernel, two readers: CCC 0.92, with 84.3% of features reproducible (CCC ≥ 0.85). **Different kernels: CCC 0.38, only 15.2% reproducible.**
- Texture fell from 0.88 to 0.61 across kernels and wavelet features from 0.92 to 0.35.
- A CNN kernel conversion (residual learning) brought CCC back to 0.84 and 57.4% reproducible.

## Relevance to our project
- This puts a number on the kernel problem our cohort faces (roughly 16–17% sharp-kernel scans in our NLST subset, per PROGRESS.md), and it shows texture and wavelet features are most affected. Intensity-threshold features such as calcium are also kernel-sensitive, because sharp kernels add noise that crosses the 130 HU threshold, which is why we used a noise-adaptive threshold.
