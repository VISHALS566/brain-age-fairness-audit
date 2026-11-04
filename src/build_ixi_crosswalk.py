"""
build_ixi_crosswalk.py
───────────────────────
Builds and verifies the benchmark subject-to-demographic crosswalk for the IXI benchmark.

Accounting & Provenance:
  1. Scans data/IXI/IXI_data (Train, Val, Test) for all 576 preprocessed .pkl volumes.
  2. Matches benchmark file index (subject_X.pkl -> X) to demographic records in data/IXI/IXI.xls.
  3. Identifies the 18 excluded volumes lacking valid chronological age or demographics.
  4. Associates the 558 matched records with unified experiment splits from outputs/unified/unified_manifest.csv:
     - 534 usable in experimental partitions (374 Train, 53 Val, 107 Test)
     - 24 unassigned / held out
  5. Assigns cohort partition strata:
     - Stratum A (index <= 314): N = 301 usable (218 train, 33 val, 50 test)
     - Stratum B (315 <= index <= 428): N = 108 usable (69 train, 7 val, 32 test)
     - Stratum C (index >= 429): N = 125 usable (87 train, 13 val, 25 test)
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "IXI" / "IXI_data"
XLS_PATH = BASE_DIR / "data" / "IXI" / "IXI.xls"
MANIFEST_PATH = BASE_DIR / "outputs" / "unified" / "unified_manifest.csv"
OUT_CSV = BASE_DIR / "data" / "IXI" / "ixi_subject_crosswalk.csv"


def main():
    print("=" * 80)
    print("BUILDING IXI BENCHMARK DEMOGRAPHIC CROSSWALK")
    print("=" * 80)

    # 1. Load IXI demographics
    print(f"[1/4] Loading demographic spreadsheet: {XLS_PATH.name}")
    meta_df = pd.read_csv(XLS_PATH)
    meta_df["Subject_ID"] = meta_df["Subject_ID"].astype(int)

    # 2. Scan all .pkl volumes across Train, Val, Test
    print(f"[2/4] Scanning volumes in {DATA_DIR}...")
    pkl_files = sorted(DATA_DIR.rglob("subject_*.pkl"))
    print(f"      Total raw benchmark volumes found: {len(pkl_files)}")
    assert len(pkl_files) == 576, f"Expected 576 raw benchmark volumes, found {len(pkl_files)}"

    # 3. Match each volume to demographic metadata
    print("[3/4] Matching file indices to demographic records...")
    rows = []
    excluded_count = 0

    # Load canonical manifest to assign verified splits and strata
    man_split_map = {}
    man_stratum_map = {}
    if MANIFEST_PATH.exists():
        man_df = pd.read_csv(MANIFEST_PATH)
        man_ixi = man_df[man_df["dataset"] == "IXI"]
        for _, r in man_ixi.iterrows():
            fn = Path(r["file"]).name
            man_split_map[fn] = r["split"]
            man_stratum_map[fn] = r["site"]

    for pkl_path in pkl_files:
        fn = pkl_path.name
        m = re.match(r"subject_(\d+)\.pkl", fn)
        if not m:
            continue
        sub_idx = int(m.group(1))

        sub_meta = meta_df[meta_df["Subject_ID"] == sub_idx]
        if sub_meta.empty:
            excluded_count += 1
            continue

        row_meta = sub_meta.iloc[0]
        age = row_meta.get("AGE")
        if pd.isna(age) or float(age) <= 0:
            excluded_count += 1
            continue

        sex_id = row_meta.get("SEX_ID (1=m, 2=f)")
        sex = "Male" if sex_id == 1 else "Female" if sex_id == 2 else "Unknown"

        # Stratum assignment by cutoffs
        if sub_idx <= 314:
            default_stratum = "Stratum A"
            stratum_code = 0.25
            historical_alias = "Hammersmith (Philips 3T)"
        elif sub_idx <= 428:
            default_stratum = "Stratum B"
            stratum_code = 0.50
            historical_alias = "Guy's (Philips 1.5T)"
        else:
            default_stratum = "Stratum C"
            stratum_code = 0.75
            historical_alias = "KCL/IoP (GE 1.5T)"

        assigned_stratum = man_stratum_map.get(fn, default_stratum)
        split = man_split_map.get(fn, "UNASSIGNED")

        rows.append({
            "benchmark_file_id": fn,
            "sequential_file_index": sub_idx,
            "demographic_row_index": sub_idx,
            "chronological_age": round(float(age), 2),
            "biological_sex": sex,
            "height_cm": int(row_meta.get("HEIGHT", 0)) if not pd.isna(row_meta.get("HEIGHT")) else 0,
            "weight_kg": int(row_meta.get("WEIGHT", 0)) if not pd.isna(row_meta.get("WEIGHT")) else 0,
            "study_date": str(row_meta.get("STUDY_DATE", "")),
            "assigned_stratum": assigned_stratum,
            "stratum_numeric_code": stratum_code,
            "split": split,
            "historical_reference_alias": historical_alias,
            "provenance_notes": "Preprocessed FreeSurfer benchmark volume from Junyu Chen et al. (2022). Sequential file index matches demographic row index in IXI.xls."
        })

    crosswalk_df = pd.DataFrame(rows)
    print(f"      Matched demographic records: {len(crosswalk_df)}")
    print(f"      Excluded volumes (missing age/demographics): {excluded_count}")
    assert len(crosswalk_df) == 558, f"Expected 558 matched records, got {len(crosswalk_df)}"
    assert excluded_count == 18, f"Expected 18 excluded records, got {excluded_count}"

    usable_df = crosswalk_df[crosswalk_df["split"] != "UNASSIGNED"]
    print(f"      Usable in experimental partitions: {len(usable_df)}")
    print(f"      Unassigned / held-out: {len(crosswalk_df) - len(usable_df)}")
    assert len(usable_df) == 534, f"Expected 534 usable records, got {len(usable_df)}"

    print("\n[4/4] Split counts in crosswalk:")
    print(crosswalk_df["split"].value_counts())
    print("\nStratum counts in usable partitions (N=534):")
    print(usable_df["assigned_stratum"].value_counts())

    # Save verified crosswalk
    crosswalk_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved verified crosswalk -> {OUT_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
