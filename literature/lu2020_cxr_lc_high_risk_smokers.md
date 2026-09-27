# Deep Learning Using Chest Radiographs to Identify High-Risk Smokers for Lung Cancer Screening Computed Tomography: Development and Validation of a Prediction Model

- **Authors:** Lu MT, Raghu VK, Mayrhofer T, Aerts HJWL, Hoffmann U
- **Year / venue:** 2020, *Annals of Internal Medicine* 173(9):704–713
- **DOI:** 10.7326/M20-1868 · PMID 32866413 · PMCID PMC9200444
- **How verified:** PubMed record and abstract.

## Key findings
- CXR-LC is a CNN on the chest radiograph plus age, sex and current smoking (yes/no). It was developed in PLCO (n = 41,856), validated in PLCO smokers (n = 5,615, 12-yr follow-up) and **NLST (n = 5,493, 6-yr follow-up)**.
- PLCO AUC 0.755 vs 0.634 for CMS eligibility. It matched PLCOm2012 (11 inputs, including smoking duration/intensity): 0.755 vs 0.751.
- **In NLST: CXR-LC AUC 0.659 vs PLCOm2012 0.650.** Discrimination is low in this pre-selected heavy-smoker group.
- Takeaway: an image plus minimal EMR data can stand in for detailed smoking history.

## Relevance to our project
- This sets the realistic ceiling for non-nodule risk prediction in NLST. Even PLCOm2012 with full smoking history reaches only about 0.65, so our clinical C of 0.647 (age/sex/smoking status only) is already near that, and **ΔC of +0.01 to +0.02 is a meaningful gain in this population**.
- It shows that imaging can partly **recover smoking-history information**, which supports our hypothesis that CT calcium and thymus proxy cumulative smoking.
