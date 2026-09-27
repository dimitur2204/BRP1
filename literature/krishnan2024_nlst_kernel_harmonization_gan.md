# Lung CT harmonization of paired reconstruction kernel images using generative adversarial networks

- **Authors:** Krishnan AR, Xu K, Li TZ, Remedios LW, Sandler KL, Maldonado F, Landman BA
- **Year / venue:** 2024, *Medical Physics* 51(8):5510–5523
- **DOI:** 10.1002/mp.17028 · PMID 38530135 · PMCID PMC11321937
- **How verified:** PubMed record and abstract.

## Key findings
- Uses NLST: 1,000 same-session scan pairs, each with one soft-tissue and one hard-kernel reconstruction, across 5 vendor-specific kernel pairs (200 each).
- A pix2pix GAN per pair type and direction (10 models) was trained on 100 pairs and tested on 100.
- Conversion improved RMSE, PSNR and SSIM in all pair types and both directions (P < 0.05). It improved agreement for **percent emphysema, skeletal-muscle area and subcutaneous fat area**, and made standard radiomic features reproducible against the ground-truth kernel.

## Relevance to our project
- This is NLST-specific evidence that **many NLST sessions have both a soft and a sharp reconstruction**. Where a sharp-kernel scan is our only series, a soft-kernel series from the same session may exist in IDC. Choosing the soft series (as DeepCAC2 did) is simpler than GAN conversion and would reduce kernel noise in our calcium and fat measurements.
- These paired reconstructions could also directly test how kernel-robust our calcium proxy is: compute it on both reconstructions of the same session.
