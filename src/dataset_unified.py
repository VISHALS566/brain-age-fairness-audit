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

    # Subject-ID partition convention (001-314 Hammersmith, 315-428 Guy's, 429+ KCL)
    # As documented in Section 3.1, these labels represent preprocessed cohort partition strata
    # rather than isolated scanner hardware physics.
    def _site(sid: int) -> str:
        if sid <= 314:  return "Hammersmith"
        if sid <= 428:  return "Guys"
        return "KCL"

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
      - Healthy controls (CDR 0): stratified across 4 age quantiles; 15% (N=47)
        sampled using random_state=42 as held-out test subjects.
      - Cognitively impaired subjects (CDR 0.5-2, N=94): sorted by age,
        half (N=47) allocated to training and half (N=47) held out as clinical test.
    """
    df = df.copy()
    healthy = df[df.get("cdr", 0.0) == 0.0]
    impaired = df[df.get("cdr", 0.0) > 0.0]

    hb = pd.qcut(healthy["age"], q=4, labels=False, duplicates="drop")
    test_ids = []
    for b in hb.unique():
        group = healthy[hb == b]
        n = int(round(test_frac * len(group)))
        test_ids += group.sample(n, random_state=seed)["subject_id"].tolist()

    df["split"] = "TRAIN"
    df.loc[df["subject_id"].isin(test_ids), "split"] = "TEST"

    if len(impaired):
        sorted_impaired = impaired.sort_values("age")["subject_id"].tolist()
        clinical_test_ids = sorted_impaired[1::2]
        df.loc[df["subject_id"].isin(clinical_test_ids), "split"] = "CLINICAL"

    return df


def build_oasis_manifest(manifest_csv: str) -> pd.DataFrame:
    """
    Read the OASIS-1 manifest already produced by oasis_brain_age_v2 and
    re-map columns to the unified schema.
    """
    raw = pd.read_csv(manifest_csv)

    # Map SPLIT values: CV→TRAIN, Test→TEST, Clinical_Test→CLINICAL
    split_map = {"CV": "TRAIN", "Test": "TEST", "Clinical_Test": "CLINICAL",
                 "Excluded": "EXCLUDED"}

    rows = []
    for _, r in raw.iterrows():
        sp = split_map.get(r["SPLIT"], "EXCLUDED")
        if sp == "EXCLUDED":
            continue
        rows.append({
            "subject_id":   r["SUBJECT_ID"],
            "file":         r["FILE_PATH"],
            "file_type":    "hdr",
            "age":          float(r["AGE"]),
            "sex":          r["SEX"],
            "site":         "WashU",
            "dataset":      "OASIS",
            "ethnicity_id": np.nan,
            "cdr":          float(r["CDR"]) if not pd.isna(r.get("CDR")) else np.nan,
            "mmse":         float(r["MMSE"]) if not pd.isna(r.get("MMSE")) else np.nan,
            "split":        sp,
        })

    df = pd.DataFrame(rows)
    print(f"[OASIS] {len(df)} subjects  |  splits: {df.split.value_counts().to_dict()}")
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
        print(f"✅ Unified manifest saved → {save_path}")
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
        "WashU": 0.0, "Hammersmith": 0.25, "Guys": 0.5, "KCL": 0.75
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
