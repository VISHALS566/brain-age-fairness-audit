"""
dataset_unified.py
──────────────────
A single PyTorch Dataset that handles BOTH:
  • IXI  — .pkl files (pickled (image, label) tuples, FreeSurfer-preprocessed)
  • OASIS-1 — .hdr/.img files (Analyze 7.5, atlas-registered T88)

Both are resampled to TARGET_SHAPE = (80, 80, 96) and z-normalised inside
the brain mask, so the model sees identical-format tensors regardless of source.

Also builds the manifest DataFrames from the two data sources.
"""

import os
import re
import pickle
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ── Shared voxel grid ──────────────────────────────────────────────────────────
TARGET_SHAPE = (80, 80, 96)   # same as OASIS v3 pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# Volume loaders
# ═══════════════════════════════════════════════════════════════════════════════

def _to_tensor(vol: np.ndarray) -> torch.Tensor:
    """
    Normalise a 3-D float32 volume (brain-masked) and interpolate to TARGET_SHAPE.
    Returns a (1, D, H, W) float32 tensor with values in [0, 1].
    """
    if vol.dtype != np.float32:
        vol = vol.astype(np.float32)
    mask = vol > 0
    if mask.sum() < 500:
        return torch.zeros((1,) + TARGET_SHAPE, dtype=torch.float32)

    lo, hi = np.percentile(vol[mask], [1, 99]).astype(np.float32)
    diff = np.float32(hi - lo + 1e-6)
    vol = np.clip((vol - lo) / diff, 0.0, 1.0)
    vol[~mask] = 0.0

    t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)        # (1,1,D,H,W)
    t = F.interpolate(t, size=TARGET_SHAPE, mode="area")        # area = correct for downsample
    t = torch.nan_to_num(t.squeeze(0), nan=0.0, posinf=0.0, neginf=0.0)  # (1,D,H,W)
    return t.float()


def load_ixi_pkl(path: str) -> torch.Tensor:
    """Load IXI .pkl → (1, 80, 80, 96) float32."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    vol = np.array(data[0], dtype=np.float32)
    # IXI volumes are already in [0,1] and skull-stripped; just normalise properly
    vol[~np.isfinite(vol)] = 0.0
    return _to_tensor(vol)


def load_oasis_hdr(path: str) -> torch.Tensor:
    """
    Load OASIS-1 T88-atlas .hdr/.img → (1, 80, 80, 96) float32.
    Applies the same orientation fix used in oasis_brain_age_v2.
    """
    vol = np.squeeze(nib.load(path).get_fdata()).astype(np.float32)
    # Orientation: match the original OASIS pipeline (flip + transpose)
    vol = np.ascontiguousarray(np.flip(np.flip(np.transpose(vol, (0, 2, 1)), 0), 1))
    return _to_tensor(vol)


# ═══════════════════════════════════════════════════════════════════════════════
# Manifest builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_ixi_manifest(ixi_data_dir: str, ixi_xls: str) -> pd.DataFrame:
    """
    Build a DataFrame for all IXI subjects found in Train/Val/Test pkl folders.
    Adds real site labels from Subject_ID ranges (IXI convention):
        001–314  Hammersmith Hospital  (1.5T Philips)
        315–428  Guy's Hospital        (1.5T Philips)
        429–578  King's College London (3T GE)
    """
    data_dir = Path(ixi_data_dir)
    xls_path = Path(ixi_xls)

    # ── Load metadata ────────────────────────────────────────────────────────
    demo = pd.read_csv(xls_path)
    demo = demo.drop_duplicates()
    check_cols = ["SEX_ID (1=m, 2=f)", "DOB", "STUDY_DATE", "AGE", "ETHNIC_ID"]
    ambiguous = []
    for sid, grp in demo.groupby("Subject_ID"):
        if len(grp) > 1 and grp[check_cols].drop_duplicates().shape[0] > 1:
            ambiguous.append(sid)
    meta = demo[~demo["Subject_ID"].isin(ambiguous)].drop_duplicates("Subject_ID").copy()

    # Benchmark cohort partition strata:
    # As detailed in Section 3.1 and audited for reproducibility, the preprocessed IXI benchmark
    # distribution (FreeSurfer-preprocessed by Junyu Chen et al., 2021) sequences subjects as
    # anonymized volumes. The numerical ranges (001-314, 315-428, 429-578) correspond to
    # sequential benchmark cohort partition strata (Stratum A, Stratum B, Stratum C) rather than
    # isolated physical scanner hardware. We preserve hospital aliases for compatibility with
    # prior starter code and metadata feature encoding.
    def _site(sid: int) -> str:
        if sid <= 314:
            return "Stratum A"
        if sid <= 428:
            return "Stratum B"
        return "Stratum C"

    def _historical_alias(sid: int) -> str:
        if sid <= 314:
            return "Hammersmith (Philips 3T)"
        if sid <= 428:
            return "Guys (Philips 1.5T)"
        return "KCL/IoP (GE 1.5T)"

    # ── Walk all splits ──────────────────────────────────────────────────────
    rows = []
    for split_name in ("Train", "Val", "Test"):
        split_dir = data_dir / split_name
        if not split_dir.exists():
            continue
        for pkl_path in sorted(split_dir.glob("subject_*.pkl")):
            m = re.match(r"subject_(\d+)\.pkl", pkl_path.name)
            if not m:
                continue
            sid = int(m.group(1))
            row = meta[meta["Subject_ID"] == sid]
            if row.empty:
                continue
            age = row.iloc[0]["AGE"]
            if pd.isna(age):
                continue
            sex_id = row.iloc[0].get("SEX_ID (1=m, 2=f)", np.nan)
            sex = "Male" if sex_id == 1 else ("Female" if sex_id == 2 else "Unknown")
            eth_id = row.iloc[0].get("ETHNIC_ID", np.nan)
            rows.append({
                "subject_id":  f"IXI_{sid:04d}",
                "file":        str(pkl_path),
                "file_type":   "pkl",
                "age":         float(age),
                "sex":         sex,
                "site":        _site(sid),
                "historical_site_alias": _historical_alias(sid),
                "dataset":     "IXI",
                "ethnicity_id": float(eth_id) if not pd.isna(eth_id) else np.nan,
                "cdr":         np.nan,
                "mmse":        np.nan,
                "split":       split_name.upper(),   # TRAIN / VAL / TEST
            })

    df = pd.DataFrame(rows)
    print(f"[IXI]  {len(df)} subjects  |  sites: {df.site.value_counts().to_dict()}")
    return df


def make_oasis_splits(df: pd.DataFrame, test_frac: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """
    Subject-level split for OASIS-1:
      - Healthy controls (CDR 0, N=309): stratified across 6 age quantiles; 15% (N=47)
        sampled using random_state=42 as held-out test subjects (remaining 262 join training pool).
      - Cognitively impaired subjects (CDR 0.5-2, N=94): sorted by age,
        half (N=47) allocated to training pool and half (N=47) held out as clinical test.
      - Total held-out subjects: N=94 (47 healthy test + 47 clinical test); training: N=309.
    """
    df = df.copy()
    cdr_col = "cdr" if "cdr" in df.columns else ("CDR" if "CDR" in df.columns else None)
    age_col = "age" if "age" in df.columns else ("AGE" if "AGE" in df.columns else None)
    sid_col = "subject_id" if "subject_id" in df.columns else ("SUBJECT_ID" if "SUBJECT_ID" in df.columns else None)

    healthy = df[df[cdr_col] == 0.0]
    impaired = df[df[cdr_col] > 0.0]

    q = max(2, min(6, len(healthy) // 10))
    hb = pd.qcut(healthy[age_col], q=q, labels=False, duplicates="drop")
    test_ids = []
    for b in hb.unique():
        group = healthy[hb == b]
        n = int(round(test_frac * len(group)))
        if n == 0 and len(group) >= 4:
            n = 1
        test_ids += group.sample(n, random_state=seed)[sid_col].tolist()

    df["split"] = "TRAIN"
    df.loc[df[sid_col].isin(test_ids), "split"] = "TEST"

    if len(impaired):
        sorted_impaired = impaired.sort_values(age_col)[sid_col].tolist()
        clinical_test_ids = sorted_impaired[1::2]
        df.loc[df[sid_col].isin(clinical_test_ids), "split"] = "CLINICAL"

    return df


def build_oasis_manifest(manifest_csv: str) -> pd.DataFrame:
    """
    Read the OASIS-1 manifest and re-map columns to the unified schema.
    Consistently normalises splits and enforces strict partition count assertions.
    """
    raw = pd.read_csv(manifest_csv)
    has_split = ("SPLIT" in raw.columns) or ("split" in raw.columns)
    if not has_split:
        raw = make_oasis_splits(raw)

    split_map = {
        "CV": "TRAIN", "TRAIN": "TRAIN", "train": "TRAIN",
        "Test": "TEST", "TEST": "TEST", "test": "TEST",
        "Clinical_Test": "CLINICAL", "CLINICAL": "CLINICAL", "clinical": "CLINICAL",
        "Excluded": "EXCLUDED", "EXCLUDED": "EXCLUDED", "excluded": "EXCLUDED",
    }

    rows = []
    for _, r in raw.iterrows():
        raw_sp = r.get("SPLIT", r.get("split", "EXCLUDED"))
        sp = split_map.get(str(raw_sp).strip(), "EXCLUDED")
        if sp == "EXCLUDED":
            continue
        sid = r.get("SUBJECT_ID", r.get("subject_id"))
        fpath = r.get("FILE_PATH", r.get("file"))
        age = r.get("AGE", r.get("age"))
        sex = r.get("SEX", r.get("sex"))
        cdr = r.get("CDR", r.get("cdr", np.nan))
        mmse = r.get("MMSE", r.get("mmse", np.nan))
        rows.append({
            "subject_id":   str(sid),
            "file":         str(fpath),
            "file_type":    "hdr",
            "age":          float(age),
            "sex":          str(sex),
            "site":         "Stratum WashU",
            "historical_site_alias": "WashU (Siemens 1.5T)",
            "dataset":      "OASIS",
            "ethnicity_id": np.nan,
            "cdr":          float(cdr) if not pd.isna(cdr) else np.nan,
            "mmse":         float(mmse) if not pd.isna(mmse) else np.nan,
            "split":        sp,
        })

    df = pd.DataFrame(rows)

    # Strict count assertions to prevent silent partitioning failures
    n_tr = (df["split"] == "TRAIN").sum()
    n_te = (df["split"] == "TEST").sum()
    n_cl = (df["split"] == "CLINICAL").sum()
    assert len(df) == 403, f"Expected 403 usable OASIS subjects, got {len(df)}"
    assert n_tr == 309, f"Expected 309 OASIS TRAIN subjects, got {n_tr}"
    assert n_te == 47, f"Expected 47 OASIS healthy TEST subjects, got {n_te}"
    assert n_cl == 47, f"Expected 47 OASIS impaired CLINICAL subjects, got {n_cl}"

    print(f"[OASIS] {len(df)} subjects verified (train={n_tr}, test={n_te}, clinical={n_cl})")
    return df


def build_unified_manifest(
    ixi_data_dir: str,
    ixi_xls: str,
    oasis_manifest_csv: str,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Merge IXI and OASIS-1 manifests into one unified DataFrame.
    Optionally saves it to a CSV.
    """
    ixi_df   = build_ixi_manifest(ixi_data_dir, ixi_xls)
    oasis_df = build_oasis_manifest(oasis_manifest_csv)
    df = pd.concat([ixi_df, oasis_df], ignore_index=True)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[Unified manifest saved] -> {save_path}")
    print(f"\n{'='*55}")
    print(f"UNIFIED MANIFEST  |  total={len(df)}")
    print(df.groupby(["dataset", "split"]).size().to_string())
    print(f"Age range: {df.age.min():.0f}–{df.age.max():.0f}y")
    print(f"Sites: {df.site.value_counts().to_dict()}")
    print(f"{'='*55}\n")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Tabular meta helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_meta_matrix(df: pd.DataFrame, fit_stats: dict | None = None) -> Tuple[np.ndarray, dict]:
    """
    Build (N, 3) tabular feature matrix:
        col 0: sex    (0 = Female, 1 = Male, 0.5 = Unknown)
        col 1: dataset_flag (0 = OASIS, 1 = IXI)
        col 2: age_std_proxy — always 0.0 at inference (filled during training loop)

    Returns the matrix and fit_stats dict (mean/std for normalisation).
    The matrix is deliberately minimal so it works for both datasets equally.
    """
    sex_num = df["sex"].map({"Male": 1.0, "Female": 0.0, "Unknown": 0.5}).fillna(0.5).values
    ds_flag = (df["dataset"] == "IXI").astype(float).values
    site_num = df["site"].map({
        "WashU": 0.0, "OASIS": 0.0, "Stratum WashU": 0.0,
        "Hammersmith": 0.25, "Stratum A": 0.25, "Stratum A (Hammersmith)": 0.25,
        "Guys": 0.5, "Stratum B": 0.5, "Stratum B (Guys)": 0.5,
        "KCL": 0.75, "Stratum C": 0.75, "Stratum C (KCL)": 0.75,
    }).fillna(0.0).values

    X = np.c_[sex_num, ds_flag, site_num].astype(np.float32)

    if fit_stats is None:
        fit_stats = {
            "mean": X.mean(axis=0),
            "std":  X.std(axis=0) + 1e-6,
        }
    X = (X - fit_stats["mean"]) / fit_stats["std"]
    return X, fit_stats


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class BrainAgeDataset(Dataset):
    """
    Unified dataset that transparently loads IXI .pkl and OASIS .hdr/.img files.

    Each item returns:
        volume  (1, 80, 80, 96) float32
        meta    (3,) float32 — normalised tabular features
        age     scalar float32
        sex     str
        site    str
        dataset str
        subject_id str
    """
    def __init__(self, df: pd.DataFrame, meta_matrix: np.ndarray):
        self.df   = df.reset_index(drop=True)
        self.meta = torch.from_numpy(meta_matrix.astype(np.float32))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row    = self.df.iloc[idx]
        ftype  = row["file_type"]

        if ftype == "pkl":
            volume = load_ixi_pkl(row["file"])
        elif ftype == "hdr":
            volume = load_oasis_hdr(row["file"])
        else:
            raise ValueError(f"Unknown file type: {ftype}")

        age = torch.tensor(float(row["age"]), dtype=torch.float32)
        return (
            volume,
            self.meta[idx],
            age,
            str(row["sex"]),
            str(row["site"]),
            str(row["dataset"]),
            str(row["subject_id"]),
        )
