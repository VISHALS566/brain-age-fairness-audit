"""
model_v2.py
───────────
Improved 3-D ResNet for brain-age regression.

Key upgrades over Small3DCNN:
  • GroupNorm instead of BatchNorm  → stable with small batches (2-8)
  • Residual (skip) connections     → deeper without vanishing gradients
  • Tabular meta-data branch        → sex, nWBV/eTIV (OASIS) or site flag (IXI)
  • Soft-label age head             → predict distribution over 2-year bins,
                                      age = expected value (reduces regression-to-mean)
  • ~1 M parameters (not 7 M)      → won't overfit on ~700 subjects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Age-bin config (shared with training script) ───────────────────────────────
AGE_LO  = 14.0   # youngest expected subject
AGE_HI  = 100.0  # oldest expected subject
BIN_W   = 2.0    # 2-year bins
BIN_CENTERS = torch.arange(AGE_LO + BIN_W / 2, AGE_HI, BIN_W)   # (K,)
K_BINS  = len(BIN_CENTERS)


def decode(logits: torch.Tensor) -> torch.Tensor:
    """Soft-label decode: expected age from logits (B, K) → (B,)."""
    return (F.softmax(logits.float(), dim=1) * BIN_CENTERS.to(logits.device)).sum(1)


def soft_targets(y: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Gaussian soft label around true age y (B,) → (B, K)."""
    d = (BIN_CENTERS.to(y.device)[None] - y[:, None]) / sigma
    t = torch.exp(-0.5 * d * d)
    return t / t.sum(1, keepdim=True)


# ── Building blocks ────────────────────────────────────────────────────────────
def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with at most 8 groups (safe for any channel count)."""
    return nn.GroupNorm(min(8, channels), channels)


class ResBlock(nn.Module):
    """Pre-activation residual block with GroupNorm."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1   = _gn(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2   = _gn(out_ch)
        # Skip connection (only needed when shape changes)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                _gn(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.gn1(self.conv1(x)), inplace=True)
        y = self.gn2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


# ── Main model ─────────────────────────────────────────────────────────────────
class BrainAgeResNet(nn.Module):
    """
    3-D ResNet brain-age predictor with an optional tabular meta branch.

    Args
    ────
    in_ch    : number of image channels (1 = T1 only)
    n_meta   : number of tabular features (sex, nWBV, eTIV …)
    n_bins   : number of soft-label age bins (= K_BINS)
    dropout  : dropout probability in the regression head
    """
    def __init__(
        self,
        in_ch:   int   = 1,
        n_meta:  int   = 3,   # 0 = image-only, >0 = tabular features (sex, site, dataset)
        n_bins:  int   = K_BINS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_meta = n_meta

        # ── Image stream ────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, stride=2, padding=1, bias=False),
            _gn(16),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResBlock(16,  32, stride=2),
            ResBlock(32,  64, stride=2),
            ResBlock(64, 128, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)   # → (B, 128, 1, 1, 1)

        # ── Tabular meta stream (disabled if n_meta == 0) ───────────────────
        if n_meta > 0:
            self.meta_branch = nn.Sequential(
                nn.Linear(n_meta, 16),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 3),
            )
            head_in = 128 + 16
        else:
            self.meta_branch = None
            head_in = 128

        # ── Regression head ─────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, n_bins),      # logits over age bins
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor = None) -> torch.Tensor:
        """
        x    : (B, in_ch, D, H, W)  float32
        meta : (B, n_meta)          float32 (optional if n_meta == 0)
        returns logits (B, n_bins) — use decode() to get predicted age in years
        """
        img_feat = self.pool(self.body(self.stem(x))).flatten(1)   # (B, 128)
        if self.n_meta > 0 and meta is not None:
            meta_feat = self.meta_branch(meta)                      # (B, 16)
            fused = torch.cat([img_feat, meta_feat], dim=1)         # (B, 144)
        else:
            fused = img_feat                                        # (B, 128)
        return self.head(fused)                                     # (B, n_bins)


def build_model(in_ch: int = 1, n_meta: int = 3) -> BrainAgeResNet:
    m = BrainAgeResNet(in_ch=in_ch, n_meta=n_meta, n_bins=K_BINS)
    total = sum(p.numel() for p in m.parameters())
    print(f"[BrainAgeResNet]  |  params: {total:,}  |  in_ch={in_ch}  |  n_meta={n_meta}  |  bins={K_BINS}")
    return m
