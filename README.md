# Fairness Audit of Brain-Age Prediction Across Sex, Cohort Strata and Age

[![Conference](https://img.shields.io/badge/Conference-CARES%202027-blue.svg)](https://cares2027.org)
[![Proceedings](https://img.shields.io/badge/Proceedings-Springer%20LNEE-green.svg)](https://www.springer.com/series/7818)
[![Track](https://img.shields.io/badge/Special%20Session-01%3A%20Multimedia%20E--Health-orange.svg)](https://cares2027.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2BCUDA-red.svg)](https://pytorch.org/)

**Official Code, Manifest, and Evaluation Repository** for the paper:
> **Fairness Audit of Brain-Age Prediction Across Sex, Cohort Strata and Age: Subgroup Calibration Results and a Proposed Reporting Checklist**  
> *Vishal S and Pranav Eashwaran G*  
> School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, India  
> `vishal.s2024a@vitstudent.ac.in`, `pranaveashwaran.g2024@vitstudent.ac.in`  
> *Target Venue: CARES 2027: International Conference on Computing and Artificial Intelligence for E-health Systems (Springer LNEE)*

---

## 📌 Abstract

The brain-age gap ($\text{BAG} = \hat{y} - y$), the difference between MRI-predicted and chronological age, is widely used as a biomarker for neurodegeneration, yet its calibration across demographic and cohort subgroups is rarely reported. We present a descriptive subgroup audit of 3D residual brain-age networks trained on 937 T1-weighted scans pooled from OASIS-1 (one cohort, $N=403$) and the preprocessed IXI benchmark (three sequential partition strata that are not verified acquisition sites, $N=534$).

We compare an **image-only network** (MAE 7.579 y, 95% CI [6.82, 8.42], $R^2 = 0.787$) with a **metadata-assisted network** that also receives sex, dataset, and cohort stratum indicators (MAE 6.938 y, 95% CI [6.18, 7.77], $R^2 = 0.813$) on a held-out test cohort ($N = 201$). Metadata lowered global error by 0.641 y but was associated with larger observed subgroup gaps in this run: the sex gap (male minus female MAE) widened from +0.463 y to +0.920 y and the Stratum A minus Stratum B gap widened from +0.594 y to +2.478 y; bootstrap intervals overlap, indicating a single-run descriptive observation rather than an established general effect.

A deployable inverse age calibration (no test-time use of age) is reported separately from an evaluation-only retrospective residual adjustment. On OASIS-1, mean BAG rose from −1.29 y (CDR 0) to +2.88 y (CDR 0.5) and +4.16 y (CDR 1) and correlated with MMSE (Pearson $r = −0.272, p = 0.025, n = 68$). We propose a 10-point reporting checklist derived from the audit, apply it to this study, and treat it as exploratory.

---

## 📊 Canonical Experimental Results

### 1. Cohort Composition and Partitions (Table 1)

| Cohort Stratum | Dataset Source | Usable $N$ | Train $N$ | Val $N$ | Test $N$ |
|:---|:---|:---:|:---:|:---:|:---:|
| **Stratum WashU** | OASIS-1 | 403 | 309 | 0 | 94 |
| **Stratum A** | IXI Benchmark | 301 | 218 | 33 | 50 |
| **Stratum B** | IXI Benchmark | 108 | 69 | 7 | 32 |
| **Stratum C** | IXI Benchmark | 125 | 87 | 13 | 25 |
| **Total** | **Pooled Cohort** | **937** | **683** | **53** | **201** |

*Note on OASIS-1 training partition: Contains 261 healthy (CDR 0) and 48 impaired (CDR 0.5–1) subjects ($261 + 48 = 309$). Held-out test cohort contains 47 healthy and 47 impaired subjects ($N = 94$).*

### 2. Subgroup Performance & Disparity Audit (Table 2)

Evaluated on the pooled held-out test cohort ($N=201$ subjects, 1,000 bootstrap iterations; IXI $N=107$, OASIS-1 $N=94$):

| Attribute | Subgroup Value | $N$ | Image-Only ResNet [95% CI] | Metadata-Assisted ResNet [95% CI] | Difference ($\Delta$) |
|:---|:---|:---:|:---:|:---:|:---:|
| **Overall** | All Subjects | 201 | 7.579 y [6.82, 8.42] | **6.938 y** [6.18, 7.77] | −0.641 y |
| **Sex** | Female | 110 | 7.369 y [6.18, 8.51] | **6.522 y** [5.42, 7.65] | −0.848 y |
| **Sex** | Male | 91 | 7.832 y [6.73, 8.99] | **7.442 y** [6.27, 8.73] | −0.390 y |
| **Cohort Stratum** | Stratum B | 32 | 7.366 y [5.35, 9.43] | **5.800 y** [4.45, 7.22] | −1.566 y |
| **Cohort Stratum** | Stratum A | 50 | 7.960 y [6.33, 9.72] | **8.278 y** [6.51, 10.21] | +0.318 y |
| **Cohort Stratum** | Stratum C | 25 | 6.798 y [4.53, 9.35] | **7.067 y** [5.39, 8.91] | +0.269 y |
| **Cohort Stratum** | Stratum WashU | 94 | 7.656 y [6.57, 8.77] | **6.579 y** [5.36, 7.88] | −1.077 y |
| **Dataset** | IXI (Normative) | 107 | 7.511 y [6.36, 8.70] | **7.254 y** [6.18, 8.32] | −0.257 y |
| **Dataset** | OASIS-1 (Clinical) | 94 | 7.656 y [6.55, 8.78] | **6.579 y** [5.45, 7.91] | −1.077 y |
| **Subgroup Gap** | **Male − Female Gap** | — | **+0.463 y** | **+0.920 y** | *Widened* |
| **Subgroup Gap** | **Stratum A − Stratum B Gap** | — | **+0.594 y** | **+2.478 y** | *Widened* |

### 3. Post-Processing & Calibration Strategies (Table 3)

Source: `outputs/phase1_fixes/postprocessing_comparison.csv` (generated via live `fit_gpr_postprocess.py` with 5-fold cross-validation and fold-specific PCA/scaling):

| Stage / Method | Inference Deployment | Target-Age Leakage | Test MAE | Test $R^2$ | Bias | Within $\pm 5$ y | Within $\pm 10$ y |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Small 3D CNN (Dev Baseline)** | Yes | None | 15.600 y | −0.190 | — | 8.4% | ~30% |
| **Image-Only 3D ResNet** | Yes | None | 7.579 y | 0.787 | +0.491 y | 40.8% | 72.6% |
| **Primary 3D ResNet (CNN)** | Yes | None | 6.938 y | 0.813 | +0.441 y | 47.8% | 75.1% |
| **Standalone GPR (50 PCA dims)** | Yes | None | 7.752 y | 0.780 | +0.211 y | 40.3% | 71.1% |
| **Prespecified CNN+GPR Blend (0.79/0.21)** | Yes | None | 6.961 y | 0.813 | +0.393 y | 46.8% | 73.6% |
| **Deployable Inverse Calibration** ($\frac{\hat{y} - 3.895}{0.933}$) | **Yes** | **None (Zero)** | **7.218 y** | **0.798** | **+0.229 y** | **44.8%** | **73.6%** |
| **Retrospective Residual Adjustment** ($\hat{y} - (a y + b)$) | No (Eval only) | Uses test $y$ | 6.756 y | 0.827 | −0.000 y | 45.3% | 78.1% |

*Note: Row 1 reflects the exploratory single-epoch development baseline evaluated strictly on IXI test subjects ($N=107$). Rows 2–7 reflect unified models evaluated on the held-out test cohort ($N=201$). Calibration parameters ($\alpha = 0.9330, \beta = 3.8952$) are fitted on in-sample CNN training predictions.*

### 4. Biological & Clinical Validity on OASIS-1 (Table 4)

- **CDR 0.0 (Healthy Controls, $N=47$):** Mean BAG = **$-1.29\text{ y}$**, Median = **$-0.12\text{ y}$**, MAE = 7.18 y
- **CDR 0.5 (Very Mild Impairment, $N=32$):** Mean BAG = **$+2.88\text{ y}$**, Median = **$+3.64\text{ y}$**, MAE = 6.02 y
- **CDR 1.0 (Mild Impairment, $N=13$):** Mean BAG = **$+4.16\text{ y}$**, Median = **$+3.10\text{ y}$**, MAE = 5.66 y
- **CDR 2.0 (Moderate Impairment, $N=2$):** Mean BAG = **$-0.88\text{ y}$**, Median = **$-0.88\text{ y}$**, MAE = 7.32 y *(small sample)*
- **Statistical Tests:**
  - Kruskal-Wallis test across CDR 0, 0.5, 1.0: $H = 5.97, p = 0.051$ (borderline at $\alpha = 0.05$)
  - Mann-Whitney $U$ test (healthy vs. impaired): $U = 790.0, p = 0.018$
  - Cognitive MMSE Correlation: Pearson $r = -0.272, p = 0.025$ ($n = 68$)

---

## 📁 Repository Structure

```text
├── CHECKLIST.md              # 10-Point Brain-Age Prediction Reporting Checklist
├── LICENSE                   # MIT License
├── README.md                 # Project documentation and reproduction instructions
├── paper.tex                 # Complete LaTeX manuscript (Springer LLNCS template, 10 pages)
├── llncs.cls                 # Official Springer LNCS document class
├── splncs04.bst              # Official Springer LNCS BibTeX style
├── figures/                  # Clean publication figures (Fig 1 to Fig 6, RGB, no alpha)
│   ├── fig1_cohort.png
│   ├── fig2_training.png
│   ├── fig3_scatter.png
│   ├── fig4_subgroups.png
│   ├── fig5_decades.png
│   ├── fig6a_cdr.png
│   └── fig6b_mmse.png
├── data/
│   └── IXI/
│       ├── README.md         # IXI volume accounting and provenance documentation
│       └── ixi_subject_crosswalk.csv # Benchmark file-index to demographic subject crosswalk
├── src/                      # Source code
│   ├── build_ixi_crosswalk.py # Standalone script to build & verify IXI crosswalk
│   ├── dataset_unified.py    # Multi-site loader & OASIS split generator
│   ├── model_v2.py           # 3D ResNet (Image-Only and Metadata-Assisted)
│   ├── train_unified.py      # Metadata-assisted model training
│   ├── train_unified_image_only.py # Image-only ablation model training
│   ├── fit_gpr_postprocess.py # PCA + GPR blending & calibration regression fitting
│   ├── evaluate_calibration_and_bootstrap.py # Bootstrap CIs & deployable calibration
│   ├── compare_models.py     # Disparity audit comparing image-only vs metadata
│   ├── plot_image_vs_metadata.py # Publication figure generation
│   ├── extract_oasis.py      # OASIS-1 dataset extraction utility
│   └── evaluate_fairness.py  # Stratified subgroup metrics
└── outputs/                  # Numerical predictions and publication figures
    ├── comparison/           # Side-by-side CSV audit metrics
    ├── figures/              # Publication figures
    ├── image_only/           # Image-only predictions, fairness report, checkpoint
    ├── phase1_fixes/         # Postprocessing comparison, bootstrap CIs, CDR/MMSE stats
    └── unified/              # Metadata model predictions, history, checkpoint, cache
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation

```bash
git clone https://github.com/VISHALS566/brain-age-fairness-audit.git
cd brain-age-fairness-audit
pip install -r requirements.txt
```

### 2. Verify Crosswalk & Subject Accounting

```bash
python src/build_ixi_crosswalk.py
```

### 3. Reproduce Post-Processing & Calibration in ~5 Seconds

```bash
python src/fit_gpr_postprocess.py
```
*Note: Latent features are pre-extracted and cryptographically validated via SHA256 checksums (`outputs/unified/latent_features_cache.npz`). If the model checkpoint or manifest is modified, the cache is automatically invalidated and re-extracted.*

---

## 📄 License & Citation

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
