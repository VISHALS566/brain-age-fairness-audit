"""
train_unified.py
────────────────
Unified brain-age training script for OASIS-1 + IXI combined dataset.

What this does
──────────────
1. Builds a unified manifest (OASIS-1 + IXI) with consistent columns
2. Trains a BrainAgeResNet with:
   - Soft-label age head (KL-divergence + SmoothL1)
   - Age-balanced sampling (50/50 decade balance)
   - Warmup + cosine LR schedule
   - EMA weights (better generalisation than last epoch)
   - Flip test-time augmentation
3. Runs fairness audit on test set:
   - Stratify BAG by: sex, site, dataset, age decade
   - Mann-Whitney U tests for group differences
4. Saves:
   - best_model_unified.pt  (EMA weights)
   - unified_predictions.csv
   - fairness_report.csv
   - training_history.csv

Usage
─────
  # From D:\\brainage directory:
  python src/train_unified.py [--epochs 50] [--batch_size 4] [--lr 1e-3] [--workers 2]
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats
from sklearn.metrics import mean_absolute_error, r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

warnings.filterwarnings("ignore")

# ── Make src importable when called from project root ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dataset_unified import (
    BrainAgeDataset,
    build_meta_matrix,
    build_unified_manifest,
    TARGET_SHAPE,
)
from model_v2 import (
    BrainAgeResNet,
    K_BINS,
    BIN_CENTERS,
    decode,
    soft_targets,
    build_model,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Paths  (edit if your data lives elsewhere)
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR          = Path(__file__).resolve().parent.parent   # D:\brainage
OUT_DIR           = BASE_DIR / "outputs" / "unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)
UNIFIED_MANIFEST  = OUT_DIR / "unified_manifest.csv"

IXI_DATA_DIR      = BASE_DIR / "data" / "IXI" / "IXI_data"
IXI_XLS           = BASE_DIR / "data" / "IXI" / "IXI.xls"
OASIS_MANIFEST    = BASE_DIR / "data" / "OASIS-1" / "manifest.csv"
if not OASIS_MANIFEST.exists():
    alt_manifest = Path("E:/code/outputs/brain_age_v2/manifest.csv")
    if alt_manifest.exists():
        OASIS_MANIFEST = alt_manifest

SEED = 42

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_age_balanced_sampler(ages: np.ndarray) -> WeightedRandomSampler:
    """
    Over-sample under-represented age decades so the model sees a balanced
    age distribution per batch → reduces regression-to-the-mean.
    """
    decade = (ages // 10).astype(int)
    counts = np.bincount(decade)
    weight_per_decade = 1.0 / np.sqrt(counts.clip(min=1))
    sample_weights = weight_per_decade[decade]
    sample_weights /= sample_weights.sum()
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(ages),
        replacement=True,
    )


def augment(x: torch.Tensor, lr_dim: int = 4) -> torch.Tensor:
    """
    Lightweight on-GPU augmentation for a batch (B,1,D,H,W):
      • random left-right flip
      • small intensity jitter (gain + noise, brain voxels only)
    """
    B, dev = x.size(0), x.device
    # LR flip
    flip = (torch.rand(B, device=dev) < 0.5).view(B, 1, 1, 1, 1)
    x = torch.where(flip, x.flip(lr_dim), x)
    # Intensity jitter
    gain  = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=dev) * 2 - 1) * 0.10
    noise = torch.randn_like(x) * 0.02 * (x > 0)
    return (x * gain + noise).clamp(0.0, 1.0)


@torch.no_grad()
def predict_ema(ema_model: nn.Module,
                loader: DataLoader,
                device: torch.device) -> tuple[np.ndarray, np.ndarray, list, list, list, list]:
    """
    Run inference with EMA model + flip TTA.
    Returns: (true_ages, pred_ages, sexes, sites, datasets, subject_ids)
    """
    ema_model.eval()
    true_ages, pred_ages = [], []
    sexes, sites, datasets, sids = [], [], [], []

    for batch in loader:
        vol, meta, age, sex, site, ds, sid = batch
        vol  = vol.to(device)
        meta = meta.to(device)

        logits  = ema_model(vol, meta)
        logits_f = ema_model(vol.flip(4), meta)          # flip along W-axis
        preds   = 0.5 * (decode(logits) + decode(logits_f))

        true_ages.extend(age.numpy().tolist())
        pred_ages.extend(preds.cpu().numpy().tolist())
        sexes.extend(list(sex))
        sites.extend(list(site))
        datasets.extend(list(ds))
        sids.extend(list(sid))

    return (
        np.array(true_ages, dtype=np.float32),
        np.array(pred_ages, dtype=np.float32),
        sexes, sites, datasets, sids,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    seed_everything()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"\n⚙️  Device : {device}  |  AMP: {use_amp}")
    if device.type == "cuda":
        print(f"   GPU    : {torch.cuda.get_device_name(0)}")

    # ── 1. Build unified manifest ────────────────────────────────────────────
    print("\n📂 Building unified manifest …")
    df = build_unified_manifest(
        ixi_data_dir=str(IXI_DATA_DIR),
        ixi_xls=str(IXI_XLS),
        oasis_manifest_csv=str(OASIS_MANIFEST),
        save_path=str(UNIFIED_MANIFEST),
    )

    # ── 2. Splits ────────────────────────────────────────────────────────────
    # Training pool = IXI TRAIN + OASIS TRAIN(CV)
    # Validation    = IXI VAL   + OASIS TEST (healthy only — used to pick nothing, just monitor)
    # Test (fairness arm)    = IXI TEST
    # Test (clinical arm)    = OASIS CLINICAL + OASIS TEST
    train_df = df[df["split"] == "TRAIN"].reset_index(drop=True)
    val_df   = df[(df["split"] == "VAL") & (df["dataset"] == "IXI")].reset_index(drop=True)
    # For the test phase — keep IXI and OASIS separate for reporting
    test_ixi_df  = df[(df["split"] == "TEST")     & (df["dataset"] == "IXI")].reset_index(drop=True)
    test_oasis_df= df[(df["split"].isin(["TEST","CLINICAL"])) & (df["dataset"] == "OASIS")].reset_index(drop=True)

    print(f"\nSplits  →  train={len(train_df)}  val={len(val_df)}  "
          f"test_IXI={len(test_ixi_df)}  test_OASIS={len(test_oasis_df)}")

    # ── 3. Tabular meta ──────────────────────────────────────────────────────
    # Fit scaling statistics exclusively on training partition
    tr_meta, fit_stats = build_meta_matrix(train_df)
    va_meta, _ = build_meta_matrix(val_df, fit_stats=fit_stats)
    ti_meta, _ = build_meta_matrix(test_ixi_df, fit_stats=fit_stats)
    to_meta, _ = build_meta_matrix(test_oasis_df, fit_stats=fit_stats)

    json.dump(
        {"mean": fit_stats["mean"].tolist(), "std": fit_stats["std"].tolist()},
        open(OUT_DIR / "meta_stats.json", "w"),
    )

    # ── 4. Datasets + loaders ────────────────────────────────────────────────
    train_ds = BrainAgeDataset(train_df, tr_meta)
    val_ds   = BrainAgeDataset(val_df,   va_meta)
    test_ixi_ds   = BrainAgeDataset(test_ixi_df,   ti_meta)
    test_oasis_ds = BrainAgeDataset(test_oasis_df, to_meta)

    sampler  = make_age_balanced_sampler(train_df["age"].values)
    nw       = args.workers

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, num_workers=nw,
        pin_memory=use_amp, persistent_workers=(nw > 0),
    )
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=nw)
    test_ixi_loader   = DataLoader(test_ixi_ds,   batch_size=args.batch_size, shuffle=False, num_workers=nw)
    test_oasis_loader = DataLoader(test_oasis_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    # ── 5. Model, EMA, optimizer ─────────────────────────────────────────────
    model = build_model(in_ch=1, n_meta=3).to(device)
    ema   = copy.deepcopy(model).to(device).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    def lr_lambda(ep):
        warmup = args.warmup
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(1, args.epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    EMA_DECAY  = 0.99
    L1_WEIGHT  = 0.10    # weight of auxiliary SmoothL1 term
    SOFT_SIGMA = 2.0     # Gaussian width in years

    # z-score stats for the auxiliary SmoothL1 term
    AGE_MEAN = float(train_df["age"].mean())
    AGE_STD  = float(train_df["age"].std() + 1e-6)

    PRINT_EVERY = 20   # print progress every N batches within an epoch
    total_batches = math.ceil(len(train_ds) / args.batch_size)

    print(f"\n🚀 Training for {args.epochs} epochs  |  batch={args.batch_size}  |  lr={args.lr}", flush=True)
    print(f"   Train N={len(train_df)}  (OASIS+IXI combined)", flush=True)
    print(f"   Age mean/std in training pool: {AGE_MEAN:.1f} / {AGE_STD:.1f}", flush=True)
    print(f"   Batches per epoch: ~{total_batches}  (each batch loads {args.batch_size} MRI volumes from disk)", flush=True)
    print(f"   ⏳ Epoch 1 will take longest — volumes load cold from disk (~1-3 min). Later epochs are faster.\n", flush=True)

    history = []
    best_val_mae = float("inf")
    step = 0

    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        seen = 0
        batch_num = 0

        print(f"--- Epoch {ep+1}/{args.epochs}  [loading first batch…]", flush=True)

        for batch in train_loader:
            batch_num += 1
            vol, meta, age, *_ = batch
            vol  = augment(vol.to(device))
            meta = meta.to(device)
            y    = age.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(vol, meta)

            logits = logits.float()
            # KL loss (soft-label distribution matching)
            loss_kl = F.kl_div(
                F.log_softmax(logits, dim=1),
                soft_targets(y, sigma=SOFT_SIGMA),
                reduction="batchmean",
            )
            # Auxiliary SmoothL1 on expected age (stabilises early training)
            pred_age = decode(logits)
            loss_l1  = F.smooth_l1_loss(
                (pred_age - AGE_MEAN) / AGE_STD,
                (y        - AGE_MEAN) / AGE_STD,
                beta=0.5,
            )
            loss = loss_kl + L1_WEIGHT * loss_l1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            decay = min(EMA_DECAY, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
            step += 1

            running_loss += loss.item() * len(y)
            seen         += len(y)

            # ── Live batch progress ──────────────────────────────────────────
            if batch_num % PRINT_EVERY == 0 or batch_num == total_batches:
                elapsed_so_far = time.time() - t0
                eta = elapsed_so_far / batch_num * (total_batches - batch_num)
                print(
                    f"  Ep {ep+1:2d} | batch {batch_num:3d}/{total_batches} "
                    f"| loss {running_loss/max(seen,1):.4f} "
                    f"| {elapsed_so_far:.0f}s elapsed "
                    f"| ETA {eta:.0f}s",
                    flush=True,
                )

        scheduler.step()

        # ── Validation (EMA, no TTA for speed) ──────────────────────────────
        ema.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                vol, meta, age, *_ = batch
                vol  = vol.to(device)
                meta = meta.to(device)
                val_preds.extend(decode(ema(vol, meta)).cpu().numpy().tolist())
                val_true.extend(age.numpy().tolist())

        val_mae = mean_absolute_error(val_true, val_preds)
        elapsed = time.time() - t0

        rec = {
            "epoch":      ep + 1,
            "train_loss": running_loss / max(seen, 1),
            "val_mae":    val_mae,
            "lr":         optimizer.param_groups[0]["lr"],
        }
        history.append(rec)

        marker = ""
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(ema.state_dict(), OUT_DIR / "best_model_unified.pt")
            marker = "  ✅ best"

        print(
            f"✔ Ep {ep+1:3d}/{args.epochs}  "
            f"loss {rec['train_loss']:.4f}  "
            f"val MAE {val_mae:.2f}y  "
            f"best {best_val_mae:.2f}y  "
            f"lr {rec['lr']:.2e}  "
            f"{elapsed:.0f}s{marker}",
            flush=True,
        )

    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    print(f"\n✅ Training done. Best val MAE: {best_val_mae:.2f}y")

    # ── 6. Test evaluation (load best EMA) ───────────────────────────────────
    print("\n🔍 Running test evaluation with best EMA model + flip TTA …")
    ema.load_state_dict(torch.load(OUT_DIR / "best_model_unified.pt", map_location=device))

    def run_test(loader, label):
        true, pred, sx, si, ds, sid = predict_ema(ema, loader, device)
        mae  = mean_absolute_error(true, pred)
        r2   = r2_score(true, pred)
        bias = float(np.mean(pred - true))
        w5   = float(np.mean(np.abs(pred - true) <= 5) * 100)
        w10  = float(np.mean(np.abs(pred - true) <= 10) * 100)
        print(f"\n── {label} ──")
        print(f"   N={len(true)}  MAE={mae:.2f}y  R²={r2:.3f}  Bias={bias:+.2f}y  "
              f"Within5y={w5:.1f}%  Within10y={w10:.1f}%")
        return pd.DataFrame({
            "subject_id": sid, "age": true, "prediction": pred,
            "bag": pred - true, "abs_error": np.abs(pred - true),
            "sex": sx, "site": si, "dataset": ds,
        })

    pred_ixi_df   = run_test(test_ixi_loader,   "IXI TEST (fairness arm)")
    pred_oasis_df = run_test(test_oasis_loader,  "OASIS TEST (clinical arm)")

    pred_all = pd.concat([pred_ixi_df, pred_oasis_df], ignore_index=True)
    pred_all.to_csv(OUT_DIR / "unified_predictions.csv", index=False)

    # ── 7. Fairness audit ─────────────────────────────────────────────────────
    print("\n📊 Fairness audit …")
    fairness_rows = []

    def group_metrics(sub: pd.DataFrame, group_name: str, group_val: str) -> dict:
        y = sub["age"].values
        p = sub["prediction"].values
        e = p - y
        ss_res = np.sum((y - p)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            "group_type":  group_name,
            "group_value": group_val,
            "N":           len(sub),
            "MAE":         float(np.mean(np.abs(e))),
            "RMSE":        float(np.sqrt(np.mean(e**2))),
            "R2":          float(r2),
            "mean_BAG":    float(np.mean(e)),
            "std_BAG":     float(np.std(e)),
            "Within5y_%":  float(np.mean(np.abs(e) <= 5) * 100),
            "Within10y_%": float(np.mean(np.abs(e) <= 10) * 100),
        }

    # Overall
    fairness_rows.append(group_metrics(pred_all, "Overall", "All"))

    # By sex
    for name, sub in pred_all.groupby("sex"):
        fairness_rows.append(group_metrics(sub, "Sex", str(name)))

    # By site
    for name, sub in pred_all.groupby("site"):
        fairness_rows.append(group_metrics(sub, "Site", str(name)))

    # By dataset
    for name, sub in pred_all.groupby("dataset"):
        fairness_rows.append(group_metrics(sub, "Dataset", str(name)))

    # By age decade
    pred_all["age_decade"] = (pred_all["age"] // 10 * 10).astype(int).astype(str) + "s"
    for name, sub in pred_all.groupby("age_decade"):
        fairness_rows.append(group_metrics(sub, "Age_decade", str(name)))

    # Statistical tests: sex
    if pred_all["sex"].nunique() == 2:
        sexes = pred_all["sex"].unique()
        g1 = pred_all[pred_all["sex"] == sexes[0]]["abs_error"].values
        g2 = pred_all[pred_all["sex"] == sexes[1]]["abs_error"].values
        if len(g1) >= 5 and len(g2) >= 5:
            stat, pval = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
            print(f"\n  Mann-Whitney U (sex, |error|): p={pval:.4f} "
                  f"→ {'⚠️ significant bias' if pval < 0.05 else '✅ no significant sex bias'}")

    # Statistical tests: site (Kruskal-Wallis)
    site_groups = [g["abs_error"].values for _, g in pred_all.groupby("site") if len(g) >= 5]
    if len(site_groups) >= 2:
        stat, pval = scipy_stats.kruskal(*site_groups)
        print(f"  Kruskal-Wallis (site, |error|): p={pval:.4f} "
              f"→ {'⚠️ significant site bias' if pval < 0.05 else '✅ no significant site bias'}")

    fairness_df = pd.DataFrame(fairness_rows)
    fairness_df.to_csv(OUT_DIR / "fairness_report.csv", index=False)

    print("\n" + "="*65)
    print(fairness_df[["group_type","group_value","N","MAE","mean_BAG","std_BAG","Within5y_%"]].to_string(index=False))
    print("="*65)
    print(f"\n📁 All outputs saved to: {OUT_DIR}")
    print(f"   • best_model_unified.pt")
    print(f"   • unified_manifest.csv")
    print(f"   • unified_predictions.csv")
    print(f"   • fairness_report.csv")
    print(f"   • training_history.csv")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified OASIS+IXI brain-age trainer")
    parser.add_argument("--epochs",     type=int,   default=50,   help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=4,    help="Batch size (4 fits RTX 4050 6GB)")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--warmup",     type=int,   default=5,    help="Warmup epochs")
    parser.add_argument("--workers",    type=int,   default=0,    help="DataLoader workers (0 for safe memory on Windows)")
    args = parser.parse_args()
    train(args)
