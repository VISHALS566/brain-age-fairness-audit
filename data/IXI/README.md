# IXI Preprocessed Benchmark Dataset & Demographic Crosswalk

## Volume Accounting & Split Provenance

This document provides transparent, reproducible provenance for the preprocessed IXI benchmark volumes used in this brain-age fairness audit.

### 1. Raw Benchmark Volumes vs. Matched Demographic Records
- **Total Volumes in Raw Benchmark:** The preprocessed FreeSurfer-aligned benchmark collection from Junyu Chen et al. (2021) originally contained **576** sequential volume files (`subject_0.pkl` to `subject_578.pkl`).
- **Demographic Exclusions ($N = 18$):** Exactly 18 volumes were excluded due to missing chronological age, unresolvable demographic rows, or corrupt records in the clinical metadata spreadsheet (`IXI.xls`).
- **Matched Records ($N = 558$):** Exactly 558 benchmark volumes were successfully mapped to verified demographic rows (`IXI.xls`) with valid biological sex, chronological age, and acquisition metadata.

### 2. Experimental Split Partitions ($N = 534$ Usable)
Of the 558 matched subjects, **534** met all volume quality criteria and were partitioned into the canonical experimental splits used in this study:
- **Training Partition:** $N = 374$
- **Validation Partition:** $N = 53$ (used strictly for epoch checkpoint selection)
- **Held-Out Test Partition:** $N = 107$
- **Unassigned / Held-Out Volumes:** $N = 24$ (held out from training/validation/test)

### 3. Cohort Partition Strata (Neutral Benchmark Indexing)
Scans in the preprocessed benchmark are identified by sequential anonymized filenames. Historical literature informally grouped subject ID cutoffs by London hospital collection centres. To avoid confounding physical scanner hardware with benchmark partition strata, our audit treats these groups as benchmark cohort strata:
- **Stratum A** ($N = 301$ usable; 218 train, 33 val, 50 test; code = 0.25; historical alias Hammersmith)
- **Stratum B** ($N = 108$ usable; 69 train, 7 val, 32 test; code = 0.50; historical alias Guy's)
- **Stratum C** ($N = 125$ usable; 87 train, 13 val, 25 test; code = 0.75; historical alias KCL/IoP)
- **Total IXI Usable:** $N = 534$

### 4. Crosswalk Schema (`ixi_subject_crosswalk.csv`)
- `benchmark_file_id`: Filename in the benchmark directory (`subject_X.pkl`).
- `sequential_file_index`: Integer identifier extracted from filename ($X$).
- `demographic_row_index`: 0-indexed row number in `IXI.xls`.
- `chronological_age`: Chronological age in years from demographic record.
- `biological_sex`: Biological sex (`Female` or `Male`).
- `height_cm`: Subject height in centimeters.
- `weight_kg`: Subject weight in kilograms.
- `study_date`: Acquisition study date from metadata.
- `assigned_stratum`: Benchmark cohort stratum (`Stratum A`, `Stratum B`, `Stratum C`).
- `stratum_numeric_code`: Normalized feature representation (`0.25`, `0.50`, `0.75`).
- `split`: Canonical partition assignment (`TRAIN`, `VAL`, `TEST`, `UNASSIGNED`).
- `historical_reference_alias`: Historical collection centre alias for provenance.
- `provenance_notes`: Processing details and pipeline origin.
