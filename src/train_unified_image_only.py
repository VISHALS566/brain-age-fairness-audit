"""
train_unified_image_only.py
───────────────────────────
Ablation Study: Image-Only BrainAgeResNet (n_meta = 0).
Trained strictly on 3D T1 MRI volumes WITHOUT tabular demographic metadata
(no sex, no site code, no dataset flag).

Directly addresses Reviewer/Professor Critique #1:
Provides true image-only fairness benchmarks across Sex, Site, and Age subgroups.
"""

import argparse
import copy
import gc
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
sys.path.insert(0, str(Path(__file__).parent))

from dataset_unified import BrainAgeDataset, build_meta_matrix, TARGET_SHAPE
from model_v2 import BrainAgeResNet, K_BINS, BIN_CENTERS, decode, soft_targets, build_model

BASE_DIR = Path(__file__).resolve().parent.parent
UNIFIED_MANIFEST = BASE_DIR / "outputs" / "unified" / "unified_manifest.csv"
OUT_DIR = BASE_DIR / "outputs" / "image_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_age_balanced_sampler(ages: np.ndarray) -> WeightedRandomSampler:
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
    B, dev = x.size(0), x.device
    flip = (torch.rand(B, device=dev) < 0.5).view(B, 1, 1, 1, 1)
    x = torch.where(flip, x.flip(lr_dim), x)
    gain = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=dev) * 2 - 1) * 0.10
    noise = torch.randn_like(x) * 0.02 * (x > 0)
    return (x * gain + noise).clamp(0.0, 1.0)

@torch.no_grad()
def predict_ema_image_only(ema_model: nn.Module, loader: DataLoader, device: torch.device):
    ema_model.eval()
    true_ages, pred_ages = [], []
    sexes, sites, datasets, sids = [], [], [], []

    for batch in loader:
        vol, meta, age, sex, site, ds, sid = batch
        vol = vol.to(device)

        logits = ema_model(vol)
        logits_f = ema_model(vol.flip(4))  # flip along width axis
        preds = 0.5 * (decode(logits) + decode(logits_f))

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

def train_image_only(args):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"\n[IMAGE-ONLY ABLATION] Device: {device} | AMP: {use_amp}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load the exact same unified manifest
    print(f"Loading unified manifest from {UNIFIED_MANIFEST}...")
    df = pd.read_csv(UNIFIED_MANIFEST)

    train_df = df[df["split"] == "TRAIN"].reset_index(drop=True)
    val_df   = df[(df["split"] == "VAL") & (df["dataset"] == "IXI")].reset_index(drop=True)
    test_ixi_df   = df[(df["split"] == "TEST") & (df["dataset"] == "IXI")].reset_index(drop=True)
    test_oasis_df = df[(df["split"].isin(["TEST", "CLINICAL"])) & (df["dataset"] == "OASIS")].reset_index(drop=True)

    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test_IXI={len(test_ixi_df)}, Test_OASIS={len(test_oasis_df)}")

    # Dummy meta matrix: fit scaling statistics exclusively on training partition
    tr_meta, fit_stats = build_meta_matrix(train_df)
    va_meta, _ = build_meta_matrix(val_df, fit_stats=fit_stats)
    ti_meta, _ = build_meta_matrix(test_ixi_df, fit_stats=fit_stats)
    to_meta, _ = build_meta_matrix(test_oasis_df, fit_stats=fit_stats)

    train_ds = BrainAgeDataset(train_df, tr_meta)
    val_ds   = BrainAgeDataset(val_df,   va_meta)
    test_ixi_ds   = BrainAgeDataset(test_ixi_df,   ti_meta)
    test_oasis_ds = BrainAgeDataset(test_oasis_df, to_meta)

    sampler = make_age_balanced_sampler(train_df["age"].values)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, num_workers=0,
        pin_memory=use_amp,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_ixi_loader   = DataLoader(test_ixi_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_oasis_loader = DataLoader(test_oasis_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Build IMAGE-ONLY model (n_meta = 0)
    print("\nBuilding Image-Only 3D ResNet (n_meta=0)...")
    model = build_model(in_ch=1, n_meta=0).to(device)
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

    EMA_DECAY = 0.99
    L1_WEIGHT = 0.10
    SOFT_SIGMA = 2.0
    AGE_MEAN = float(train_df["age"].mean())
    AGE_STD  = float(train_df["age"].std() + 1e-6)

    ckpt_path = OUT_DIR / "best_model_image_only.pt"
    start_epoch = 0
    best_val_mae = float("inf")
    history = []

    if getattr(args, "resume", False) and ckpt_path.exists():
        print(f"Resuming training from checkpoint {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        ema.load_state_dict(ckpt)

    total_batches = math.ceil(len(train_ds) / args.batch_size)
    print(f"Training for epochs {start_epoch+1} to {args.epochs} | batch={args.batch_size} | lr={args.lr}")
    print(f"Batches per epoch: ~{total_batches}\n", flush=True)

    step = start_epoch * total_batches

    for ep in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        seen = 0
        batch_num = 0

        for batch in train_loader:
            batch_num += 1
            vol, _, age, *_ = batch
            vol = augment(vol.to(device))
            y   = age.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(vol)  # No meta passed!

            logits = logits.float()
            loss_kl = F.kl_div(
                F.log_softmax(logits, dim=1),
                soft_targets(y, sigma=SOFT_SIGMA),
                reduction="batchmean",
            )
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

            decay = min(EMA_DECAY, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
            step += 1

            running_loss += loss.item() * len(y)
            seen         += len(y)

            if batch_num % 30 == 0 or batch_num == total_batches:
                elapsed_so_far = time.time() - t0
                print(
                    f"  Ep {ep+1:2d}/{args.epochs:2d} | batch {batch_num:3d}/{total_batches} "
                    f"| loss {running_loss/max(seen,1):.4f} | {elapsed_so_far:.0f}s elapsed",
                    flush=True,
                )

        scheduler.step()

        # Validation (EMA)
        ema.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                vol, _, age, *_ = batch
                vol = vol.to(device)
                val_preds.extend(decode(ema(vol)).cpu().numpy().tolist())
                val_true.extend(age.numpy().tolist())

        val_mae = mean_absolute_error(val_true, val_preds)
        elapsed = time.time() - t0

        rec = {
            "epoch": ep + 1,
            "train_loss": running_loss / max(seen, 1),
            "val_mae": val_mae,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(rec)

        marker = ""
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(ema.state_dict(), OUT_DIR / "best_model_image_only.pt")
            marker = "  [BEST]"

        print(
            f"Ep {ep+1:2d}/{args.epochs:2d} | "
            f"loss {rec['train_loss']:.4f} | "
            f"val MAE {val_mae:.2f}y | "
            f"best {best_val_mae:.2f}y | "
            f"{elapsed:.0f}s{marker}",
            flush=True,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    print(f"\nTraining Complete. Best validation MAE: {best_val_mae:.2f} years")

    # 3. Test Evaluation (Load Best EMA Checkpoint)
    print("\nRunning test evaluation with best EMA Image-Only checkpoint...")
    ema.load_state_dict(torch.load(OUT_DIR / "best_model_image_only.pt", map_location=device))

    def evaluate_test_loader(loader, label):
        true, pred, sx, si, ds, sid = predict_ema_image_only(ema, loader, device)
        mae  = mean_absolute_error(true, pred)
        r2   = r2_score(true, pred)
        bias = float(np.mean(pred - true))
        w5   = float(np.mean(np.abs(pred - true) <= 5) * 100)
        w10  = float(np.mean(np.abs(pred - true) <= 10) * 100)
        print(f"[{label}] N={len(true)} | MAE={mae:.2f}y | R2={r2:.3f} | Bias={bias:+.2f}y | W5%={w5:.1f}% | W10%={w10:.1f}%")
        return pd.DataFrame({
            "subject_id": sid, "age": true, "prediction": pred,
            "bag": pred - true, "abs_error": np.abs(pred - true),
            "sex": sx, "site": si, "dataset": ds,
        })

    pred_ixi = evaluate_test_loader(test_ixi_loader, "IXI TEST (Fairness Arm)")
    pred_oasis = evaluate_test_loader(test_oasis_loader, "OASIS TEST (Clinical Arm)")

    pred_all = pd.concat([pred_ixi, pred_oasis], ignore_index=True)
    pred_all.to_csv(OUT_DIR / "image_only_predictions.csv", index=False)

    # 4. Fairness Audit for Image-Only Model
    fairness_rows = []
    def group_metrics(sub: pd.DataFrame, gtype: str, gval: str):
        y = sub["age"].values
        p = sub["prediction"].values
        e = p - y
        ss_res = np.sum((y - p)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            "group_type": gtype,
            "group_value": gval,
            "N": len(sub),
            "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e**2))),
            "R2": float(r2),
            "mean_BAG": float(np.mean(e)),
            "std_BAG": float(np.std(e)),
            "Within5y_%": float(np.mean(np.abs(e) <= 5) * 100),
            "Within10y_%": float(np.mean(np.abs(e) <= 10) * 100),
        }

    fairness_rows.append(group_metrics(pred_all, "Overall", "All"))
    for name, sub in pred_all.groupby("sex"):
        fairness_rows.append(group_metrics(sub, "Sex", str(name)))
    for name, sub in pred_all.groupby("site"):
        fairness_rows.append(group_metrics(sub, "Site", str(name)))
    for name, sub in pred_all.groupby("dataset"):
        fairness_rows.append(group_metrics(sub, "Dataset", str(name)))

    pred_all["age_decade"] = (pred_all["age"] // 10 * 10).astype(int).astype(str) + "s"
    for name, sub in pred_all.groupby("age_decade"):
        fairness_rows.append(group_metrics(sub, "Age_decade", str(name)))

    fair_df = pd.DataFrame(fairness_rows)
    fair_df.to_csv(OUT_DIR / "image_only_fairness.csv", index=False)

    print("\n" + "=" * 70)
    print("IMAGE-ONLY FAIRNESS REPORT")
    print("=" * 70)
    print(fair_df[["group_type", "group_value", "N", "MAE", "mean_BAG", "Within5y_%"]].to_string(index=False))
    print("=" * 70)
    print(f"Outputs saved to: {OUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for ablation")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    train_image_only(args)

