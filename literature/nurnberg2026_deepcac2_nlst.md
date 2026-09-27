# Coronary artery calcification assessment in National Lung Screening Trial CT images (DeepCAC2)

- **Authors:** Leonard Nürnberg, Simon Bernatz, Borek Foldyna, Michael T. Lu, Andrey Fedorov, Hugo J. W. L. Aerts
- **Year / venue:** 2026, arXiv preprint 2603.24633 (eess.IV), submitted 25 Mar 2026. Not peer-reviewed.
- **URL:** https://arxiv.org/abs/2603.24633
- **How verified:** arXiv API metadata plus the PDF full text.

## Key findings
- Released as a public dataset: automated CAC segmentations, Agatston-type CAC scores and risk categories for **127,776 NLST CT scans from 26,228 participants**, covering all time points and all reconstruction kernels.
- Model: a 3D U-Net inside a heart ROI (platipy nnU-Net cardiac segmentation), trained on 622 expert-annotated CTs and validated on 156. Input resampled to 0.7 × 0.7 × 2.5 mm, window [−135, 500] HU.
- Agreement with expert CAC on 390 NLST scans: Spearman 0.879. Weighted κ = 0.844 for four risk groups (0, 1–100, 101–300, >300).
- Clinical check: in 23,011 baseline scans (soft/standard kernels STANDARD, B50f, FC51 or C, slice ≤3 mm), the >300 group vs CAC = 0 had an all-cause mortality **HR 2.05 (1.77–2.39)**, adjusted for age, sex, smoking status and heart-disease history (C-index 0.685). Lung cancer was not analysed.
- Data release: segmentations and scores are promised on Zenodo and in NCI Imaging Data Commons. The model will come as an MHub.ai container (`mhubai/deepcac2`). At the time of the preprint only a 200-patient dashboard was public.

## Relevance to our project
- If released, this gives an **external, expert-validated CAC score for exactly our NLST scans**. We could check our density-weighted heart-mask calcium proxy against it (correlation, and whether both give the same lung-cancer HR) and test coronary-specific calcium against whole-heart calcium (valves and annulus).
- It comes from the same group as the thymus paper, which makes a thymus × CAC paper by them plausible. Our analysis should be framed as an early independent test of that link.
