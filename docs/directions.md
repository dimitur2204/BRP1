# Next direction: self-supervised 3D encoders on NLST

Written 2026-10-26 after reading three papers. Status: D1 is coded (S6/S7,
`docs/pipeline.md` §8); D2–D4 are proposals.

## What the papers contribute

| paper | what it does | what we take from it |
|---|---|---|
| **DrasCLR** (Yu et al., MedIA 2024) | Contrastive SSL on 3D lung CT patches (COPDGene, 9k scans). Patches are anchored to atlas landmarks by registration. Hard negatives are *the same anatomical location in other patients*. Convolutions are conditioned on location (Loc-CondConv). Frozen features + linear readout predict spirometry, emphysema grades and survival (Cox + confounders, C 0.76 vs 0.74 hand-crafted). | (1) A **frozen encoder + linear Cox readout** is a legitimate, cheap evaluation protocol. (2) **Anatomical location matters**: the same texture means different things in different lung regions. Their ablation drops 6–16 points when location is randomised. (3) They needed ANTs registration (2 days). We already have TotalSegmentator **lobes** as a free anatomical coordinate system. |
| **3DINO** (Xu et al., npj Digit. Med. 2025) | DINOv2 adapted to 3D: [CLS] self-distillation + masked patch-token prediction, ViT-L/16, ~100k CT/MRI/PET volumes (incl. 500 NLST patients). Frozen linear probes beat other pretrained 3D models, and gains are largest with little labelled data. | **Public weights** (Hugging Face, CC BY-NC-ND), so a SOTA general encoder can be tested on our split *without any training*. Patch tokens give a spatial feature map. |
| **MAE vs JEPA for 3D brain MRI** (Ergün et al., arXiv 2606.13315) | Controlled comparison of SSL objectives for disease *classification* (not segmentation). MAE + a high-pass spectral loss wins when the signal is high-frequency (tumour grade); variance–covariance regularisation helps multi-factor signals. Frozen probes match or beat fine-tuning on subtle, weak-signal tasks. | (1) **The pretraining objective must match the signal's structure.** Lung-cancer risk mixes high-frequency (nodules) and diffuse (emphysema, fibrosis) signal. (2) For weak signals, **try frozen before fine-tuning**. (3) Few-shot advantage over a supervised 3D ResNet, which is our situation (~700 train events). |

## Constraints from our data

- Labels: the public IDC-780 package has lung-cancer time/event, age, sex,
  race, smoking status and screening results, plus `ctab` radiologist reads
  per screen (code 60 "significant cardiovascular abnormality" confirmed on
  the heart branch). It has **no mortality, cause of death, pack-years or
  comorbidities** (needs CDAS, `docs/future_data_request_cdas.md` on the heart
  branches). **To verify on the cluster:** whether `prsn` has cancer
  location (lobe), histology and stage. Also which other `ctab` codes and
  nodule fields (lobe, diameter, attenuation) exist.
- Images: 25,861 baseline CTs on disk; 16,345 are in the cohort with lung
  masks. ~9.5k controls have no TotalSegmentator mask. They are unlabelled
  data for SSL and need no mask for pretraining.
- Our from-scratch CNN is at roughly C ≈ 0.69 (old pipeline, 5-seed
  ensemble). Fill in the new S4 baseline numbers here.

## Proposed line of work

**Question: how much of the future lung-cancer risk visible on a baseline CT
can a self-supervised encoder capture without labels, and *where in the lung*
does that risk sit?**

**D1. Frozen 3DINO probe (started).** Feed the S2 volumes through frozen
3DINO-ViT and fit a linear Cox probe (S6, S7). Compare test C, C after 1 year and
time-dependent AUC against the S4 CNN and a clinical-only Cox on the same split,
all through `s5_evaluate.py`. This is one GPU job plus minutes of CPU. Each later step
reuses this embed → probe harness.

**D2. What does the encoder see? (positive controls).** Use the same frozen
features to fit linear probes for T0 `ctab` findings: emphysema, cardiovascular
abnormality, nodule ≥ 4 mm present, and fibrosis if coded. This is the DrasCLR readout idea.
If the probe finds emphysema but not cancer risk, the encoder is fine and the
task is hard. If it finds neither, the input (masked, 2.5 mm) is the problem.

**D3. In-domain SSL on NLST.** Continue 3DINO pretraining, or train MAE with
the spectral loss, on the ~22k baseline CTs of **train-split and
unmasked/out-of-cohort patients only** (no val/test images). Then probe again with
the identical D1 protocol. This tests the MRI paper's point that the objective
should match the signal, here in lung CT, where nobody has compared objectives
for cancer *risk*.

**D4. Lobe-level risk (the novel part).** Pool patch tokens (`--save-tokens`)
per TotalSegmentator lobe. DrasCLR's location conditioning works here
without registration. Ask whether the lobe where cancer is later diagnosed
is distinguishable from the patient's other lobes at T0, and whether this still
holds in lobes **without** a T0 nodule ≥ 4 mm. A yes would mean that diffuse tissue
change predicts where the tumour will arise. Clinically, that could support
lobe-targeted follow-up. Needs: lobe labels in the S2 cache, and the
`prsn` location fields + `ctab` nodule lobes verified.

## Risks and open points

- **Licence:** 3DINO weights are CC BY-NC-ND 4.0, so non-commercial
  research only (industry-funded research is excluded) and no redistribution.
  Confirm the project qualifies before downloading.
- **Domain mismatch:** 3DINO saw unmasked, variable-spacing volumes. Our
  input is masked lungs at 2.5 mm. If D1 is weak, rerun S6 on unmasked chest
  crops before concluding.
- **Image overlap:** 500 random NLST patients' images (not labels) were in
  3DINO pretraining. This is negligible for label leakage, but it should be stated.
- **Selection:** with ~155 val events, the l2 choice and the feature-set
  choice are noisy. Report test results only for the val-chosen setting.
