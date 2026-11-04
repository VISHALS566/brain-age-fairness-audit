# Fairness Audit of Brain-Age Prediction Across Sex, Acquisition Site and Age

[![Conference](https://img.shields.io/badge/Conference-CARES%202027-blue.svg)](https://cares2027.org)
[![Proceedings](https://img.shields.io/badge/Proceedings-Springer%20LNEE-green.svg)](https://www.springer.com/series/7818)
[![Track](https://img.shields.io/badge/Special%20Session-01%3A%20Multimedia%20E--Health-orange.svg)](https://cares2027.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2BCUDA-red.svg)](https://pytorch.org/)

**Official Code and Evaluation Repository** for the paper:
> **Fairness Audit of Brain-Age Prediction Across Sex, Acquisition Site and Age: Subgroup Calibration Results and a Proposed Reporting Checklist**  
> *Vishal S and Pranav Eashwaran G*  
> School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, India  
> `vishal.s2024a@vitstudent.ac.in`, `pranaveashwaran.g2024@vitstudent.ac.in`  
> *Accepted for submission to CARES 2027: International Conference on Computing and Artificial Intelligence for E-health Systems (Springer LNEE)*

---

## 📌 Abstract

The brain-age gap ($\text{BAG} = \hat{y} - y$), the difference between MRI-predicted and chronological age, is widely used as a biomarker for neurodegeneration, yet its calibration across demographic and acquisition subgroups is rarely reported. We present a descriptive subgroup audit of 3D residual brain-age networks trained on 937 T1-weighted scans pooled from OASIS-1 (one site) and IXI (three sites). 

We compare an **image-only network** (MAE 7.58 y, 95% CI [6.82, 8.42]) with a **metadata-assisted network** that also receives sex, dataset, and site indicators (MAE 6.94 y, 95% CI [6.13, 7.75]) on a held-out test set ($N = 201$). Metadata lowered global error by 0.64 y but widened the sex gap (male minus female MAE) from +0.46 y to +0.92 y and the Hammersmith-minus-Guy’s gap from +0.59 y to +2.48 y; bootstrap intervals overlap, so this is a single-run observation, not an established effect. 

A deployable inverse age calibration (no test-time use of age) is reported separately from an evaluation-only residual adjustment. On OASIS-1, mean BAG rose from −1.29 y (CDR 0) to +2.88 y (CDR 0.5) and +4.16 y (CDR 1) and correlated with MMSE (Pearson $r = −0.272, p = 0.025$). We propose a 10-point reporting checklist derived from the audit, apply it to this study, and treat it as exploratory.

---

## 📊 Key Results Summary

### 1. Model Performance & Disparity Comparison (Table 2)

Evaluated on the held-out test partition ($N=201$ subjects, 1,000 bootstrap iterations):

| Subgroup / Attribute | Subgroup Value | $N$ | Image-Only ResNet [95% CI] | Metadata-Assisted ResNet [95% CI] | Difference ($\Delta$) |
|:---|:---|:---:|:---:|:---:|:---:|
| **Overall** | All Subjects | 201 | 7.579 y [6.82, 8.42] | **6.938 y** [6.13, 7.75] | −0.641 y |
| **Sex** | Female | 110 | 7.369 y [6.18, 8.51] | **6.522 y** [5.42, 7.65] | −0.848 y |
| **Sex** | Male | 91 | 7.832 y [6.73, 8.99] | **7.442 y** [6.27, 8.73] | −0.390 y |
| **Site** | Guy's (1.5 T) | 32 | 7.366 y [5.35, 9.43] | **5.800 y** [4.45, 7.22] | −1.566 y |
| **Site** | Hammersmith (3.0 T) | 50 | 7.960 y [6.33, 9.72] | **8.278 y** [6.51, 10.21] | +0.318 y |
| **Site** | KCL (1.5 T) | 25 | 6.798 y [4.53, 9.35] | **7.067 y** [5.39, 8.91] | +0.269 y |
| **Site** | WashU (1.5 T) | 94 | 7.656 y [6.57, 8.77] | **6.579 y** [5.36, 7.88] | −1.077 y |
| **Dataset** | IXI (Normative) | 107 | 7.511 y [6.36, 8.70] | **7.254 y** [6.18, 8.32] | −0.257 y |
| **Dataset** | OASIS-1 (Clinical) | 94 | 7.656 y [6.55, 8.78] | **6.579 y** [5.45, 7.91] | −1.077 y |
| **Disparities** | **Male − Female Gap** | — | **+0.463 y** | **+0.920 y** | *Doubled* |
| **Disparities** | **Hammersmith − Guy's** | — | **+0.594 y** | **+2.478 y** | *Quadrupled* |

### 2. Deployable vs. Evaluation Calibration (Table 3)

| Stage / Method | Inference Deployment | Target-Age Leakage | Test MAE | Test $R^2$ | Within $\pm 5$ y | Within $\pm 10$ y |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw 3D ResNet** | Yes | None | 6.938 y | 0.813 | 47.8% | 75.1% |
| **Deployable Inverse Calibrated** ($\frac{\hat{y} - 3.68}{0.933}$) | **Yes** | **None (Zero)** | **7.212 y** | **0.798** | **44.3%** | **73.6%** |
| **Retrospective Residual Adjustment** ($\text{BAG} - (a y + b)$) | No (Eval only) | Uses test $y$ | 6.732 y | 0.824 | 48.3% | 78.1% |

### 3. Biological & Clinical Validity on OASIS-1 (Table 4)

- **CDR 0.0 (Healthy Controls, $N=47$):** Mean BAG = **$-1.29\text{ y}$**
- **CDR 0.5 (Very Mild Dementia, $N=32$):** Mean BAG = **$+2.88\text{ y}$**
- **CDR 1.0 (Mild Dementia, $N=13$):** Mean BAG = **$+4.16\text{ y}$**
- **CDR 2.0 (Moderate Dementia, $N=2$):** Mean BAG = **$-0.88\text{ y}$** *(under-sampled)*
- **Statistical Significance:**
  - One-way ANOVA across CDR 0, 0.5, 1.0: $F = 3.27, p = 0.043$
  - Mann-Whitney $U$ test (CDR 0 vs. CDR $\ge 0.5$): $p = 0.018$
  - Cognitive MMSE Correlation: Pearson $r = -0.272, p = 0.025$

---

## 📁 Repository Structure

```text
├── CHECKLIST.md              # 10-Point Brain-Age Prediction Reporting Checklist
├── LICENSE                   # MIT License
├── README.md                 # Project documentation and reproduction instructions
├── requirements.txt          # Python dependencies
├── src/                      # Source code
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
    ├── figures/              # Publication figures (Fig 1 to Fig 9, 300 DPI)
    ├── image_only/           # Image-only predictions, fairness report, checkpoint
    ├── phase1_fixes/         # Bootstrap CI tables, calibration comparison, CDR/MMSE stats
    └── unified/              # Metadata model predictions, history, checkpoint
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation

```bash
git clone https://github.com/VISHALS566/brain-age-fairness-audit.git
cd brain-age-fairness-audit
pip install -r requirements.txt
```

### 2. Run GPR Blending & Deployable Calibration (Section 3.5)

```bash
python src/fit_gpr_postprocess.py
```

### 3. Run Subgroup Bootstrap CIs & Clinical Evaluation (Phase 1)

```bash
python src/evaluate_calibration_and_bootstrap.py
```

### 4. Evaluate Image-Only vs. Metadata Disparity (Phase 2)

```bash
python src/compare_models.py
python src/plot_image_vs_metadata.py
```

---

## 📜 10-Point Reporting Checklist

See [`CHECKLIST.md`](CHECKLIST.md) for full documentation of the 10-point reporting checklist proposed in Section 5.6 of the paper.

---

## 📑 Citation

If you use this code, the checklist, or findings from this study in your research, please cite:

```bibtex
@inproceedings{vishal2027fairness,
  title={Fairness Audit of Brain-Age Prediction Across Sex, Acquisition Site and Age: Subgroup Calibration Results and a Proposed Reporting Checklist},
  author={Vishal, S. and Eashwaran G, Pranav},
  booktitle={CARES 2027: International Conference on Computing and Artificial Intelligence for E-health Systems},
  series={Lecture Notes in Electrical Engineering (LNEE)},
  publisher={Springer},
  year={2027}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
