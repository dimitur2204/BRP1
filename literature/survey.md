# Literature survey: immune ageing (thymus) and vascular ageing (cardiac calcium, pericardial fat) as CT markers of lung-cancer risk in NLST

*Compiled 2026-09-27. 28 papers, one file each in this folder.*

**Verification.** Every paper was found through PubMed E-utilities, Semantic Scholar, the arXiv API or a web search. Title, authors, year, venue and DOI were confirmed against the PubMed/arXiv record. Numbers come from the abstract unless the file says the full text was read. Details that could only be taken from secondary sources are marked **UNVERIFIED** in the relevant file; this applies only to parts of Borchardt 2025. Statements labelled *interpretation* or *expectation* are my own reasoning, not published findings.

**Our starting point.** A density-weighted cardiac calcium proxy (dilated TotalSegmentator heart mask) on NLST baseline LDCT replicated as a lung-cancer-incidence marker: test HR/SD 1.32 [1.13, 1.55], adjusted for age, sex and smoking status. It adds ΔC +0.011 to +0.014 over a clinical Cox model (test C 0.647 → 0.658). We have no pack-years.

---

## 1. The thymic-health papers (Bernatz et al., *Nature* 2026)

**How the NLST score was built** ([bernatz2026_thymic_health_adults](bernatz2026_thymic_health_adults.md)):
- **Pipeline.** A U-Net localises the thymic bed (anterior mediastinal fat) and gives a centre of mass. A 3D ResNet-50 was pretrained with SwAV self-supervision on thymic patches. A shallow classifier was then fine-tuned on expert Araki-style grades. The output is P(thymus not fully fatty).
- **Development data.** 5,674 external CTs. NLST (n = 25,031, T0) and FHS (n = 2,581) were locked test sets.
- **Percentiling.** The raw output was **percentile-ranked within each cohort** (0–100), with low ≤25, average 25–75 and high >75. Scans that fail automatic QC are put in "low".
- **Zenodo release.** `ID, Thymic_health_continuous, Thymic_health_categories` (0/1/2).
- **Not public.** Model weights are not released. The segmenter weights are public.
- **Harmonisation.** The authors explicitly did *not* address scanner or kernel batch effects and warn that no universal cut-offs exist.

**Lung-cancer incidence.**
- High vs low: **HR 0.64 (0.53–0.76)**; average vs low: HR 0.78. Both are *unadjusted*.
- After sex/age strata plus **smoking status and pack-years**, the association held at **type III P = 0.036** (adjusted HRs are only in Fig. 3b).
- Within current smokers P = 0.051 and within former smokers P = 0.25, so neither was significant alone.
- Lung-cancer mortality (HR 0.52) and CVD mortality (HR 0.37) were much more robust.

**Smoking.** In FHS, smoking duration, cigarettes per day and pack-years were each negatively associated with thymic health. Thymic health was also linked to BMI, metabolic syndrome markers and inflammatory proteins (IL-6, IL-18, OSM, CXCL10/11) and to chronic CRP.

**Coronary calcium was not examined anywhere** in the main text, the 42-page supplement or the immunotherapy companion paper. No cardiac imaging features were studied.

**Immunotherapy companion paper** ([bernatz2026_thymic_health_immunotherapy](bernatz2026_thymic_health_immunotherapy.md)):
- Validates the score biologically: higher sjTRECs, higher TCR diversity and higher tumour T-cell fraction with higher thymic health.
- Shows clinical relevance: NSCLC ICI overall survival HR 0.56 high vs low.
- Also shows how adjustment can erode the effect: TRACERx DFS HR 0.62 unadjusted → 0.84 fully adjusted.

*Interpretation (back-of-envelope).* On a uniform 0–100 percentile scale the top- and bottom-quartile means are about 2.6 SD apart. If the relationship is log-linear, HR 0.64 corresponds to roughly **HR ≈ 0.84 per SD** unadjusted. The pack-years-adjusted per-SD effect is probably closer to 0.9. That is of the same order as our calcium effect (1.32/SD, which is about 0.76 inverted).

## 2. Coronary artery calcium on LDCT and lung cancer

**CAC is measurable and prognostic on NLST LDCT.**
- Chiles 2015: visual and Agatston CAC predict CHD death, HR 6.6 for Agatston >1000.
- Zeleznik 2021: DeepCAC in NLST, adjusted HR 3.87 for ASCVD death.
- DeepCAC2 (Nürnberg 2026, preprint): automated CAC for 127,776 NLST scans, ρ = 0.88 against experts, full release pending on Zenodo/IDC.
- Lessmann 2019: CAC in NLST is strongly sex-dependent (prevalence 81% in men vs 60% in women).

**CAC and cancer incidence in general populations.**
- MESA (Handy 2016): CAC >400 goes with cancer HR 1.53, COPD 2.71 and hip fracture 4.29, after adjustment including tobacco use. CAC reads as a marker of **vascular or biological ageing**.
- MESA (Dzaye 2022), lung cancer only: log(CAC+1) **HR 1.27 → 1.22 after adjustment that includes pack-years**; CAC ≥400 went from 4.87 → 3.22.
- Heinz Nixdorf Recall (Borchardt 2025): log(CAC+1) adjusted HR **1.21**; CAC ≥400 HR 4.31. But CAC alone gives AUC 0.63 and would not improve screening. According to secondary summaries (UNVERIFIED), smoking dose was in the adjustment set and the association persisted, though weaker, in never-smokers.
- CAC Consortium (Dzaye 2021): lung-cancer *mortality* SHR 1.7 for CAC ≥400 vs 0, and **stronger in people with a smoking history (SHR 2.2)**. Pack-years were not available.

**CAC as a record of smoking exposure** (Yao 2025, 182k participants):
- CAC shows a dose-response with pack-years that continues past 20 pack-years.
- It does not plateau with intensity the way blood markers do.
- CAC **stays about 19% higher more than 30 years after quitting**, the only subclinical marker that never returns to baseline.

**Is the CAC–lung-cancer link attributed to smoking?** The authors frame it as partly shared risk factors (smoking) and partly biological ageing. The quantitative evidence, the MESA pack-years adjustment, shows **only modest attenuation (about 17% on the log scale)**. So CAC is *not merely* a stand-in for self-reported pack-years, although residual confounding from noisy self-report cannot be excluded. Our HR/SD of 1.32 without pack-years sits in the range of these adjusted estimates, which is reassuring.

## 3. Thymus, atherosclerosis and cardiovascular disease; thymus and smoking

**Thymus and atherosclerosis.**
- **Walther 2026 (n = 206)** is the only direct thymus–CAC study I found. Each higher thymic grade meant **22 fewer Agatston units after age/sex adjustment (P = 0.03)**. There was no adjustment for smoking or BMI.
- Bernatz 2026: low thymic health predicts CVD mortality in NLST (high vs low HR 0.37) and FHS (HR 0.08). FHS incident CVD attenuated after smoking adjustment.

**Thymic output matters.** In Kooshesh 2023, thymectomy was followed by higher all-cause mortality (RR 2.9), more cancer (RR 2.0), fewer new T cells (lower sjTREC) and higher proinflammatory cytokines.

**Thymus and smoking or adiposity.**
- Araki 2016 (FHS, n = 2,540): a fatty thymus goes with former smoking (49%) and **more pack-years (P = 0.04)**, plus higher BMI. These are modest effects compared with age and sex.
- Sandstedt 2023 (SCAPIS, n = 1,048): fatty degeneration is independently predicted by age, male sex, BMI/abdominal obesity and low fibre intake. **Smoking was not independent.** A fatty thymus goes with fewer naïve CD8+ T cells and lower TRECs.

**Synthesis.** Thymic involution and CAC share age, male sex, cumulative smoking, adiposity and inflammation as determinants. The one direct study shows an inverse association that survives age/sex adjustment, but it has never been tested with smoking adjustment or at scale.

## 4. Epicardial and pericardial fat on CT, cancer and smoking

- **Foldyna 2024 (NLST, n = 24,090).** EAT volume and density predict all-cause and CV mortality beyond pack-years, BMI and CAC. **EAT is higher in former than current smokers and rises with pack-years.** Smoking therefore confounds it in opposite directions: quitting brings weight gain, while cumulative dose also contributes.
- **Langenbach 2025 (NLST, serial scans).** A rise in EAT density predicts lung-cancer *mortality* (HR 1.30) after adjustment for pack-years, BMI and CAC. This is the only lung-cancer-related EAT finding in NLST.
- **Gap.** I found **no study of EAT or pericardial fat and lung-cancer incidence**, nor any study of EAT with thymus.

## 5. Opportunistic multi-organ CT, imaging ageing clocks, deep-learning risk on NLST

**Lung-cancer risk from images.**
- **Sybil** (Mikhael 2023): 6-year C-index 0.75 in NLST from the image alone (1-year AUC 0.92). This is the image-only upper benchmark.
- **CXR-LC** (Lu 2020): radiograph plus age, sex and current smoking reaches AUC 0.659 in NLST, against **0.650 for PLCOm2012**, the full smoking-history model. In this population the realistic ceiling without nodule information is about 0.65, which puts our clinical C of 0.647 near it. Images can partly *replace* detailed smoking history.

**Imaging ageing clocks and mortality.**
- **CXR-Age** (Raghu 2021): 5 years of image age carries more risk than 5 years of chronological age (HR 2.26 vs 1.77).
- **CXR-risk** (Lu MT et al., *JAMA Netw Open* 2019;2:e197416, DOI 10.1001/jamanetworkopen.2019.7416, PMID 31322692) is related and was verified on PubMed. It has no separate file and is not in the table.

**Multi-organ opportunistic CT.**
- **Pickhardt 2020:** aortic calcium is the best single biomarker (5-year death AUROC 0.743), and combining organs helps (0.811).
- **Marcinkiewicz 2024:** 32 structures on NLST predict 10-year mortality (AUC 0.72), and **CAC has the highest feature importance**.
- **Xu 2023:** NLST body composition adds to death outcomes but **not to lung-cancer incidence** once PLCOm2012 factors, emphysema and CAC are adjusted for. This is a direct warning for our endpoint.

## 6. Radiomics reproducibility across kernels; ComBat

- **Scanner variability.** Mackin 2015: interscanner variability can equal between-tumour variability, with texture features worst.
- **Kernel dependence.** Choe 2019: across kernels only **15% of 702 features are reproducible** (CCC 0.38). CNN kernel conversion restores 57%.
- **NLST pairs.** Krishnan 2024: NLST sessions often include both soft and hard reconstructions. GAN conversion reduces shifts in emphysema, muscle and fat measurements.
- **Calcium and kernels.** An 2022: CAC on LDCT agrees best with dedicated calcium CT on a **smooth kernel** (ICC about 0.9), and all kernels detect it (98.5% sensitivity).
- **ComBat.** Orlhac 2019: ComBat removes protocol effects from CT radiomics (proportion of features with a significant protocol effect falls from 98% to 15%). It is only valid when batch is not confounded with outcome.

This matches our R3/R4 experience: texture was kernel-dominated and did not replicate, while simple physical features such as calcium mass survived with noise-adaptive thresholds.

---

## Table of papers

| # | File | First author, year | Venue | DOI / ID | Topic | Key number |
|---|---|---|---|---|---|---|
| 1 | bernatz2026_thymic_health_adults | Bernatz 2026 | Nature 652:986 | 10.1038/s41586-026-10242-y | 1 | LC incidence HR 0.64 (high vs low, unadj.); adj. P = 0.036 |
| 2 | bernatz2026_thymic_health_immunotherapy | Bernatz 2026 | Nature 652:995 | 10.1038/s41586-026-10243-x | 1 | NSCLC ICI OS HR 0.56; sjTREC ↑ |
| 3 | nurnberg2026_deepcac2_nlst | Nürnberg 2026 | arXiv (preprint) | arXiv:2603.24633 | 2 | 127,776 NLST scans; ρ = 0.879 vs expert |
| 4 | chiles2015_cac_nlst_mortality | Chiles 2015 | Radiology 276:82 | 10.1148/radiol.15142062 | 2 | CHD death HR 6.63 (Agatston >1000) |
| 5 | zeleznik2021_deepcac_cardiovascular_risk | Zeleznik 2021 | Nat Commun 12:715 | 10.1038/s41467-021-20966-2 | 2 | NLST ASCVD death aHR 3.87 |
| 6 | lessmann2019_cac_tac_sex_nlst | Lessmann 2019 | JACC CVI 12:1808 | 10.1016/j.jcmg.2018.10.026 | 2 | CAC prevalence 81% men vs 60% women |
| 7 | handy2016_cac_noncardiovascular_mesa | Handy 2016 | JACC CVI 9:568 | 10.1016/j.jcmg.2015.09.020 | 2 | Cancer HR 1.53 (CAC >400) |
| 8 | dzaye2022_cac_lung_colorectal_mesa | Dzaye 2022 | EHJ-CVI 23:708 | 10.1093/ehjci/jeab099 | 2 | Lung log(CAC+1) HR 1.27 → 1.22 with pack-years |
| 9 | dzaye2021_cac_lung_cancer_mortality_cacc | Dzaye 2021 | Atherosclerosis 339:48 | 10.1016/j.atherosclerosis.2021.10.007 | 2 | LC mortality SHR 1.7; smokers 2.2 |
| 10 | borchardt2025_cac_incident_lung_cancer_hnr | Borchardt 2025 | Radiol CTI 7:e240156 | 10.1148/ryct.240156 | 2 | log(CAC+1) aHR 1.21; AUC 0.63 |
| 11 | yao2025_smoking_subclinical_cv_markers | Yao 2025 | JACC 85:1018 | 10.1016/j.jacc.2024.12.032 | 2 | CAC +19% >30 yr after quitting |
| 12 | kooshesh2023_thymectomy_health_consequences | Kooshesh 2023 | NEJM 389:406 | 10.1056/NEJMoa2302892 | 3 | Mortality RR 2.9; cancer RR 2.0 |
| 13 | araki2016_normal_thymus_ct_smoking | Araki 2016 | Eur Radiol 26:15 | 10.1007/s00330-015-3796-y | 3 | Fatty thymus ↔ pack-years (P = 0.04) |
| 14 | sandstedt2023_thymus_fatty_degeneration_scapis | Sandstedt 2023 | Immun Ageing 20:45 | 10.1186/s12979-023-00371-7 | 3 | 59% fully fatty; BMI independent, smoking not |
| 15 | walther2026_thymus_morphology_cac | Walther 2026 | Biomedicines 14:883 | 10.3390/biomedicines14040883 | 3 | −22.2 Agatston per thymus grade (age/sex adj.) |
| 16 | foldyna2024_eat_nlst_cv_risk | Foldyna 2024 | Commun Med 4:44 | 10.1038/s43856-024-00475-1 | 4 | EAT higher in former smokers; ↑ with pack-years |
| 17 | langenbach2025_eat_changes_nlst_mortality | Langenbach 2025 | Radiology 314:e240473 | 10.1148/radiol.240473 | 4 | EAT density ↑ → LC mortality HR 1.30 |
| 18 | mikhael2023_sybil | Mikhael 2023 | JCO 41:2191 | 10.1200/JCO.22.01345 | 5 | NLST 6-yr C 0.75 |
| 19 | lu2020_cxr_lc_high_risk_smokers | Lu 2020 | Ann Intern Med 173:704 | 10.7326/M20-1868 | 5 | NLST AUC 0.659 vs PLCOm2012 0.650 |
| 20 | raghu2021_cxr_age_biological_age | Raghu 2021 | JACC CVI 14:2226 | 10.1016/j.jcmg.2021.01.008 | 5 | HR 2.26 per 5 yr CXR-Age |
| 21 | xu2023_ai_body_composition_nlst | Xu 2023 | Radiology 308:e222937 | 10.1148/radiol.222937 | 5 | No added value for LC incidence |
| 22 | marcinkiewicz2024_multistructure_ai_nlst | Marcinkiewicz 2024 | Radiology 312:e240541 | 10.1148/radiol.240541 | 5 | 10-yr ACM AUC 0.72; CAC top feature |
| 23 | pickhardt2020_opportunistic_ct_biomarkers | Pickhardt 2020 | Lancet Digit Health 2:e192 | 10.1016/S2589-7500(20)30025-X | 5 | Aortic Ca 5-yr death AUROC 0.743 |
| 24 | orlhac2019_combat_ct_radiomics | Orlhac 2019 | Radiology 291:53 | 10.1148/radiol.2019182023 | 6 | Protocol-affected features 98% → 15% |
| 25 | mackin2015_ct_scanner_variability_radiomics | Mackin 2015 | Invest Radiol 50:757 | 10.1097/RLI.0000000000000180 | 6 | Interscanner ≈ inter-tumour variability |
| 26 | choe2019_kernel_conversion_radiomics | Choe 2019 | Radiology 292:365 | 10.1148/radiol.2019181960 | 6 | 15% features reproducible across kernels |
| 27 | krishnan2024_nlst_kernel_harmonization_gan | Krishnan 2024 | Med Phys 51:5510 | 10.1002/mp.17028 | 6 | NLST paired kernels; emphysema/fat agreement ↑ |
| 28 | an2022_cac_ldct_reconstruction_kernels | An 2022 | Clin Imaging 83:166 | 10.1016/j.clinimag.2021.12.024 | 6 | Smooth kernel ICC ≈ 0.9 vs calcium CT |

---

## Gaps our study can fill

1. **No study models CT thymic health and cardiac calcium or fat together.** The thymus papers never looked at CAC. The CAC–cancer papers never looked at immune markers. The only thymus–CAC study (Walther 2026) had 206 clinical patients, with no smoking or BMI adjustment and no outcomes. NLST has published thymic-health scores for 25,031 participants on the same baseline scans we already process.
2. **Independence and additivity for lung-cancer incidence** of immune-ageing and vascular-ageing CT markers is untested. It is also unknown whether either adds beyond PLCOm2012-type factors or Sybil.
3. **Smoking-proxy decomposition in heavy smokers.** MESA and HNR adjusted for pack-years in mostly lighter or never-smoking populations. No study asks, in NLST (every participant ≥30 pack-years, quit ≤15 years), how much of the calcium or thymus lung-cancer signal pack-years, smoking duration and years since quitting explain.
4. **Pericardial or epicardial fat and lung-cancer incidence** is unstudied. The NLST EAT papers covered mortality only.
5. **Kernel robustness of thymic health** was not reported: the authors used population percentiles with no harmonisation. Our kernel-annotated subset can test whether thymic health differs by kernel or manufacturer.

## Implications for hypotheses

*These are expectations derived from the literature, to be pre-registered, not findings.*

**H1: thymus and calcium are correlated, but weakly after confounder adjustment.**
- **Direction:** higher thymic health goes with lower calcium, as in Walther 2026.
- **Shared drivers:** age, male sex, pack-years (Araki, Bernatz FHS, Yao), BMI and inflammation all push the two in opposite directions, which produces a negative crude correlation.
- **Expected size:** a modest crude Spearman correlation. After partialling out age, sex and smoking status, it should be weak (|partial ρ| of about 0.1 or less). NLST's restricted age range (55–74) and uniformly heavy smoking shrink between-person variance further.
- **Walther reference point:** about 77% of the thymus–CAC slope survived age/sex adjustment there, but that was a wider age range and a clinical sample.
- **Pericardial fat:** the thymus–pericardial fat correlation should be negative and mostly BMI-driven (Sandstedt, Foldyna).
- **Technical check:** our pericardial fat shell should not overlap the thymic bed (anterior mediastinal fat), which could create an artefactual correlation.

**H2: the two markers are independent and additive for lung-cancer incidence.**
- If H1 holds (weak correlation), both HRs should change little under mutual adjustment. For example, calcium HR/SD 1.3 and thymus HR/SD about 0.85–0.9 should each move by less than 10% when the other is added.
- **Expected combined gain:** ΔC over the clinical model of about +0.01–0.03.
- **Power:** our test set has 154 events and a ΔC confidence interval of about ±0.012, which is too small to show additivity by ΔC alone. We should also report likelihood-ratio tests and mutually adjusted HRs on discovery + test.
- **Design weighting:** if our subset oversamples cases, use case-cohort weighting, as Chiles 2015 and Lessmann 2019 did.
- **Outcome prior (Xu 2023):** many CT markers that predict death fail for incidence, so we should expect smaller effects for lung-cancer incidence than for mortality.
- **Landmark analysis:** a landmark analysis that excludes cancers in the first year (the Sybil lesson) protects against prevalent-nodule effects.

**H3: both markers are partly proxies of cumulative smoking.**
- **Calcium:** strongly and persistently dose-dependent on pack-years (Yao). **Thymic health:** weakly (Araki, P = 0.04) and more tied to BMI and sex (Sandstedt).
- **Prediction:** adding pack-years (and smoking duration and years since quitting, all available from NLST CDAS) should **attenuate the calcium HR modestly (about 10–30%, as in MESA's 17%), not to null**. The thymus HR should attenuate somewhat, in line with Bernatz: unadjusted HR 0.64 against a pack-years-adjusted P of only 0.036.
- **If calcium does fall to null** after pack-years in NLST, that would be a new finding: in heavy smokers, calcium acts as a smoking-dose biomarker, not an independent ageing marker.
- **Complementary tests:**
  - (a) correlate calcium and thymus with pack-years within current and former smokers;
  - (b) in former smokers, test calcium against years since quitting (it should persist) and thymus against years since quitting (partial recovery is possible);
  - (c) compare effects across smoking-related histologies (squamous/small-cell vs adenocarcinoma). This is my suggestion, not taken from the literature.

**Practical design implications.**
- Request **pack-years, BMI, smoking duration, cigarettes per day and years since quitting** from CDAS. These are the standard adjustment set in every NLST imaging-outcome paper by the thymus group.
- Use the continuous Zenodo thymic score and do not pass it through our ComBat. Test and adjust for kernel and manufacturer.
- When DeepCAC2 is released, benchmark our calcium proxy against it and separate coronary from valve or annular calcium.
- Prefer soft-kernel series where a session has both.
- Pre-specify sex interactions: sex differences run through both the calcium and thymus literatures.
