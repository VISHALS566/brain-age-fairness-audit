# 10-Point Brain-Age Prediction Reporting Checklist

This checklist is proposed in the paper:
> **Fairness Audit of Brain-Age Prediction Across Sex, Acquisition Site and Age: Subgroup Calibration Results and a Proposed Reporting Checklist**  
> *Vishal S and Pranav Eashwaran G* (CARES-2027 / Springer LNCS)

Its goal is to promote standardized, reproducible, and transparent subgroup reporting in neuroimaging-based brain-age prediction studies.

---

### The 10 Reporting Items

| # | Item Description | What to Report | Status in This Study |
|---|---|---|:---:|
| **1** | **Demographic Subgroup Metrics** | Report sample size ($N$), Mean Absolute Error ($\text{MAE}$), and mean/SD of Brain-Age Gap ($\text{BAG} = \hat{y} - y$) separately for biological sexes (Female vs. Male). | **Met** (F: 6.52y vs M: 7.44y metadata; 7.37y vs 7.83y image-only) |
| **2** | **Acquisition Site & Hardware Breakdown** | Disaggregate performance across all imaging sites, specifying scanner manufacturer, model, and magnetic field strength ($1.5\text{T}$ vs. $3.0\text{T}$). Note nested confounding. | **Met** (Table 1 & Table 2 across Guy's, Hammersmith, KCL, WashU) |
| **3** | **Age-Stratified Calibration** | Report mean BAG and MAE across age decades. Flag small bins ($N < 10$) as statistically unreliable to detect regression-to-the-mean effects. | **Met** (+6.46y in 20s to -6.34y in 50s; 10s and 90s flagged) |
| **4** | **Bias Adjustment Transparency** | Clearly distinguish inference-time deployable calibrations from retrospective evaluation-stage residual adjustments. Explicitly document any use of ground-truth test labels. | **Partly Met** (Deployable Cole/Beheshti formula reported separately; subgroup effect of calibration left for future work) |
| **5** | **Held-Out Acquisition Site Generalization** | Test model performance on an entirely unseen acquisition site that did not contribute scans during training or validation. | **Open** (All 4 available sites contributed to training in this 2-cohort design) |
| **6** | **Training Set Demographic Documentation** | Report full age, sex, and site distributions for training and validation partitions, not solely for the test set. | **Partly Met** (Partition counts reported in Fig. 1; continuous distributions tabulated for test set) |
| **7** | **Uncertainty & Confidence Intervals** | Provide 95% confidence intervals (e.g., via $\ge 1,000$ bootstrap resamples) for all reported aggregate and subgroup metrics. | **Met** (1,000-iteration bootstrap 95% CIs reported in Table 2 for all subgroups) |
| **8** | **Clinical Biological Validity** | Validate BAG against known neurodegenerative stages (e.g., Clinical Dementia Rating [CDR]) or cognitive scores (e.g., MMSE) when clinical cohorts are included. | **Met** (OASIS-1: monotonic BAG rise across CDR 0 $\to$ 0.5 $\to$ 1.0 [$p = 0.043$]; MMSE $r = -0.272, p = 0.025$) |
| **9** | **Model Architecture & Input Modality** | Explicitly state whether demographic or acquisition attributes (sex, site, scanner) are provided as model inputs, or if the model relies purely on imaging features. | **Met** (Image-only vs. metadata-assisted models ablated side by side) |
| **10** | **Open Data, Code & Subject-Level Predictions** | Make preprocessing scripts, model definitions, trained checkpoints, and subject-level predictions publicly accessible. | **Planned / Met** (Code and prediction tables released in this repository) |

---

### Self-Audit Tally

- **Met:** 6 / 10
- **Partly Met:** 2 / 10
- **Planned:** 1 / 10
- **Open:** 1 / 10
