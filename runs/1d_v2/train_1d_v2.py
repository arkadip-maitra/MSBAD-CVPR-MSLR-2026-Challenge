#!/usr/bin/env python3
"""
CVPR MSLR 2026 Track 2 — Italian Sign Language Recognition (Radar)
1D Temporal Model v2: RangeTemporalNet + Temporal Self-Attention

Improvements over v1 (80.29%):
    Architecture:
        + Temporal Self-Attention (2 transformer layers at L=12, dim=768)
          — captures global temporal dependencies that 1D CNN misses
        + Higher stochastic depth (0.15 vs 0.10)
        + ~34M params (vs ~20M in v1)

    Training:
        + Born-again from v1 checkpoints (CNN parts loaded, attention random)
        + Two-stage: Stage 1 warms up attention+head, Stage 2 full fine-tune
        + LLRD: range_enc < temporal stages < attention < head
        + Stronger KD (alpha=0.5 vs 0.6: more teacher trust)
        + Lower pseudo-label threshold (0.70 vs 0.80: more data)
        + Better augmentation: circular time shift, amplitude scaling

Usage:
    conda run -n slr python /data/slr/train_1d_v2.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_MAX_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import torch

torch.set_num_threads(4)

import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import timm
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomErasing
from tqdm import tqdm

OUT_DIR = Path("/data/slr/checkpoints_1d_v2")
V1_DIR = Path("/data/slr/checkpoints_1d")
V7_DIR = Path("/data/slr/checkpoints_v7")
V6_DIR = Path("/data/slr/checkpoints_v6")
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "training.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

CFG = {
    "data_dir": "/data/slr/track2_",
    "output_dir": str(OUT_DIR),
    "max_frames": 48,
    "n_range_bins": 256,
    "n_classes": 126,
    "channel_means": [-64.3217, -62.9119, -63.9254],
    "channel_stds": [12.5721, 11.6620, 11.7259],
    "n_folds": 5,
    "batch_size": 128,
    "num_workers": 4,
    # Two-stage training (born-again from v1)
    "stage1_epochs": 10,
    "stage1_lr": 3e-4,
    "stage2_epochs": 250,
    "stage2_lr": 5e-4,
    "weight_decay": 0.02,
    "warmup_pct": 0.08,
    "label_smoothing": 0.1,
    "early_stopping_patience": 40,
    "grad_clip_norm": 1.0,
    # LLRD
    "llrd_decay": 0.65,
    # Knowledge distillation
    "kd_alpha": 0.5,
    "kd_tau": 4.0,
    "teacher_img_size": 224,
    # Pseudo-labeling
    "pseudo_label_threshold": 0.70,
    # Augmentation
    "mixup_alpha": 0.4,
    "cutmix_alpha": 1.0,
    "cutmix_prob": 0.5,
    "time_mask_max": 8,
    "range_mask_max": 30,
    "noise_std": 0.1,
    "time_warp_lo": 0.8,
    "time_warp_hi": 1.2,
    "time_reverse_prob": 0.5,
    "channel_shuffle_prob": 0.2,
    "channel_drop_prob": 0.1,
    "random_erase_prob": 0.20,
    "circ_shift_prob": 0.3,
    "circ_shift_max": 8,
    "amp_scale_prob": 0.5,
    "amp_scale_lo": 0.8,
    "amp_scale_hi": 1.2,
    # Model (same dims as v1 for born-again compatibility)
    "range_dim": 384,
    "temp_dims": [384, 512, 768],
    "temp_kernel": 7,
    "drop_path_rate": 0.15,
    "drop_rate": 0.3,
    "ms_dropout_samples": 5,
    # Temporal attention
    "attn_heads": 8,
    "attn_layers": 2,
    "attn_dropout": 0.1,
    # Teacher model name
    "teacher_model_name": "convnext_tiny.fb_in22k_ft_in1k",
    "pretrained": True,
    "seed": 42,
}

# ===========================================================================
#  DATA
# ===========================================================================


def collect_train_samples(data_dir):
    train_dir = os.path.join(data_dir, "train")
    samples, idx_to_cls = [], {}
    for cls_folder in sorted(os.listdir(train_dir)):
        cls_path = os.path.join(train_dir, cls_folder)
        if not os.path.isdir(cls_path):
            continue
        cls_idx = int(cls_folder.split("_", 1)[0])
        idx_to_cls[cls_idx] = cls_folder
        for sname in sorted(os.listdir(cls_path)):
            spath = os.path.join(cls_path, sname)
            if os.path.isdir(spath):
                samples.append((spath, sname, cls_idx))
    return samples, idx_to_cls


def collect_eval_samples(data_dir):
    samples = []
    for split in ("val", "test"):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for sname in sorted(os.listdir(split_dir)):
            spath = os.path.join(split_dir, sname)
            if os.path.isdir(spath):
                samples.append((spath, sname))
    return samples


def _load_one(sample_path, sample_name):
    chs = []
    for r in (1, 2, 3):
        arr = np.load(os.path.join(sample_path, f"{sample_name}_RTM{r}.npy"))
        chs.append(arr)
    return np.stack(chs, axis=0).astype(np.float32)


def preload(samples, is_test=False):
    cache = {}
    for item in tqdm(samples, desc="Pre-loading", leave=False):
        if is_test:
            spath, sname = item
        else:
            spath, sname, _ = item
        cache[sname] = _load_one(spath, sname)
    return cache


# ===========================================================================
#  DATASET
# ===========================================================================

_MEANS = np.array(CFG["channel_means"], dtype=np.float32).reshape(3, 1, 1)
_STDS = np.array(CFG["channel_stds"], dtype=np.float32).reshape(3, 1, 1)
_ERASER = RandomErasing(
    p=CFG["random_erase_prob"], scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0
)


class RadarDataset(Dataset):
    def __init__(self, keys, labels, cache, augment=False, soft_labels=None,
                 img_size=None):
        self.keys = keys
        self.labels = labels
        self.cache = cache
        self.augment = augment
        self.soft_labels = soft_labels
        self.img_size = img_size

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def _pad_or_crop(x, max_t, random_crop=False):
        t = x.shape[1]
        if t > max_t:
            s = (
                np.random.randint(0, t - max_t + 1) if random_crop
                else (t - max_t) // 2
            )
            x = x[:, s : s + max_t, :]
        elif t < max_t:
            reps = max_t // t + 1
            x = np.tile(x, (1, reps, 1))[:, :max_t, :]
        return x

    def _apply_aug(self, x):
        if np.random.random() < CFG["time_reverse_prob"]:
            x = x[:, ::-1, :].copy()
        if np.random.random() < CFG["channel_shuffle_prob"]:
            x = x[np.random.permutation(3)]
        if np.random.random() < CFG["channel_drop_prob"]:
            x[np.random.randint(0, 3)] = 0.0
        # Circular time shift
        if np.random.random() < CFG["circ_shift_prob"]:
            shift = np.random.randint(-CFG["circ_shift_max"],
                                       CFG["circ_shift_max"] + 1)
            x = np.roll(x, shift, axis=1)
        # Amplitude scaling
        if np.random.random() < CFG["amp_scale_prob"]:
            scale = np.random.uniform(CFG["amp_scale_lo"], CFG["amp_scale_hi"])
            x = x * scale
        for _ in range(2):
            if np.random.random() < 0.5:
                w = np.random.randint(1, CFG["time_mask_max"] + 1)
                t0 = np.random.randint(0, max(1, x.shape[1] - w))
                x[:, t0 : t0 + w, :] = 0.0
        for _ in range(2):
            if np.random.random() < 0.5:
                w = np.random.randint(1, CFG["range_mask_max"] + 1)
                r0 = np.random.randint(0, max(1, x.shape[2] - w))
                x[:, :, r0 : r0 + w] = 0.0
        if np.random.random() < 0.5:
            x = x + np.random.normal(0, CFG["noise_std"], x.shape).astype(
                np.float32
            )
        if np.random.random() < 0.3:
            factor = np.random.uniform(CFG["time_warp_lo"], CFG["time_warp_hi"])
            new_t = max(1, int(x.shape[1] * factor))
            x = zoom(x, (1, new_t / x.shape[1], 1), order=1).astype(np.float32)
            x = self._pad_or_crop(x, CFG["max_frames"], random_crop=True)
        return x

    def __getitem__(self, idx):
        x = self.cache[self.keys[idx]].copy()
        x = (x - _MEANS) / _STDS
        x = self._pad_or_crop(x, CFG["max_frames"], random_crop=self.augment)
        if self.augment:
            x = self._apply_aug(x)
        t = torch.from_numpy(x)
        if self.img_size is not None:
            sz = (self.img_size, self.img_size) if isinstance(self.img_size, int) else self.img_size
            t = F.interpolate(
                t.unsqueeze(0), size=sz, mode="bilinear", align_corners=False,
            ).squeeze(0)
        if self.augment:
            t = _ERASER(t)
        soft = (
            self.soft_labels[self.keys[idx]]
            if self.soft_labels else torch.zeros(CFG["n_classes"])
        )
        return t, self.labels[idx], soft


# ===========================================================================
#  1D MODEL v2 — RangeTemporalNet + Temporal Self-Attention
# ===========================================================================


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device))
        return x / keep * mask


class Conv1dBN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel, stride=stride,
            padding=kernel // 2, groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvNeXt1DBlock(nn.Module):
    def __init__(self, dim, kernel=7, expansion=4, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel, padding=kernel // 2, groups=dim, bias=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expansion, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)
        return residual + self.drop_path(x)


class MultiSampleDropout(nn.Module):
    def __init__(self, in_features, out_features, n_samples=5, drop_rate=0.3):
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(drop_rate) for _ in range(n_samples)]
        )
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        if self.training:
            return torch.mean(
                torch.stack([self.fc(d(x)) for d in self.dropouts]), dim=0
            )
        return self.fc(x)


class RangeEncoder(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            Conv1dBN(3, 64, 7, stride=2),
            Conv1dBN(64, 128, 5, stride=2),
            Conv1dBN(128, 256, 3, stride=2),
            Conv1dBN(256, embed_dim, 3, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class TemporalEncoder(nn.Module):
    def __init__(self, dims=(384, 512, 768), kernel=7, drop_path_rate=0.1):
        super().__init__()
        n_blocks = 2 + 3 + 2
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]
        i = 0

        self.stage0 = nn.Sequential(
            ConvNeXt1DBlock(dims[0], kernel, drop_path=dpr[i]),
            ConvNeXt1DBlock(dims[0], kernel, drop_path=dpr[i + 1]),
        )
        i += 2
        self.ds0 = Conv1dBN(dims[0], dims[1], 3, stride=2)

        self.stage1 = nn.Sequential(
            ConvNeXt1DBlock(dims[1], kernel, drop_path=dpr[i]),
            ConvNeXt1DBlock(dims[1], kernel, drop_path=dpr[i + 1]),
            ConvNeXt1DBlock(dims[1], kernel, drop_path=dpr[i + 2]),
        )
        i += 3
        self.ds1 = Conv1dBN(dims[1], dims[2], 3, stride=2)

        self.stage2 = nn.Sequential(
            ConvNeXt1DBlock(dims[2], kernel, drop_path=dpr[i]),
            ConvNeXt1DBlock(dims[2], kernel, drop_path=dpr[i + 1]),
        )

        self.out_dim = dims[2]

    def forward(self, x):
        x = self.stage0(x)
        x = self.ds0(x)
        x = self.stage1(x)
        x = self.ds1(x)
        x = self.stage2(x)
        return x  # (B, D, L) — no pool, let attention handle it


class TemporalAttention(nn.Module):
    """Multi-head self-attention over CNN temporal features.
    Input: (B, D, L=12) from TemporalEncoder
    Output: (B, D) after attention + global average pooling.
    """

    def __init__(self, dim=768, n_heads=8, n_layers=2, dropout=0.1, max_len=12):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.norm_in = nn.LayerNorm(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, L, D)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.norm_in(x)
        x = self.encoder(x)
        x = self.norm_out(x)
        return x.mean(dim=1)  # global average → (B, D)


class RangeTemporalNetV2(nn.Module):
    """
    v2: CNN temporal encoder + self-attention.
    Same RangeEncoder + TemporalEncoder dims as v1 for born-again compatibility.
    New TemporalAttention captures global temporal dependencies.
    """

    def __init__(self, n_classes=126, range_dim=384,
                 temp_dims=(384, 512, 768), temp_kernel=7,
                 drop_path_rate=0.15, drop_rate=0.3, ms_samples=5,
                 attn_heads=8, attn_layers=2, attn_dropout=0.1):
        super().__init__()
        self.range_enc = RangeEncoder(embed_dim=range_dim)
        self.feat_drop = nn.Dropout(0.1)
        self.temporal_enc = TemporalEncoder(
            dims=temp_dims, kernel=temp_kernel,
            drop_path_rate=drop_path_rate,
        )
        self.temporal_attn = TemporalAttention(
            dim=self.temporal_enc.out_dim, n_heads=attn_heads,
            n_layers=attn_layers, dropout=attn_dropout, max_len=12,
        )
        self.head = MultiSampleDropout(
            self.temporal_enc.out_dim, n_classes,
            n_samples=ms_samples, drop_rate=drop_rate,
        )

    def forward(self, x):
        B, C, T, R = x.shape
        frames = x.permute(0, 2, 1, 3).reshape(B * T, C, R)
        feats = self.range_enc(frames)
        feats = feats.reshape(B, T, -1).permute(0, 2, 1)  # (B, D, T)
        feats = self.feat_drop(feats)
        seq = self.temporal_enc(feats)  # (B, 768, L=12)
        temporal_feat = self.temporal_attn(seq)  # (B, 768)
        return self.head(temporal_feat)


# ===========================================================================
#  TEACHER MODEL (for loading v7/v6 checkpoints)
# ===========================================================================


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


class ConvNeXtGeM(nn.Module):
    def __init__(self, model_name, n_classes=126, pretrained=False,
                 drop_path_rate=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        n_feat = self.backbone.num_features
        self.gem = GeM(p=3.0)
        self.head = MultiSampleDropout(n_feat, n_classes, n_samples=5,
                                       drop_rate=0.3)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        return self.head(self.gem(feats).flatten(1))


# ===========================================================================
#  BORN-AGAIN from v1
# ===========================================================================


def load_born_again(model, ckpt_path, dev):
    """Load v1 checkpoint into v2 model — CNN parts loaded, attention stays random."""
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
    v1_sd = ckpt["model_state_dict"]
    model_sd = model.state_dict()

    loaded, skipped = 0, 0
    for k, v in v1_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_sd)
    val_acc = ckpt.get("val_acc", 0.0)
    log.info("  Born-again ← %s: loaded %d, skipped %d (val %.2f%%)",
             ckpt_path.name, loaded, skipped, 100 * val_acc)
    return val_acc


# ===========================================================================
#  LLRD — layer-wise learning rate decay
# ===========================================================================


def get_llrd_params(model, base_lr, decay=0.65, wd=0.02):
    """Assign different LRs to model components (deepest → lowest)."""
    groups = []

    def _add(name, params, lr_mult):
        ps = [p for p in params if p.requires_grad]
        if not ps:
            return
        no_wd = []
        with_wd = []
        for p in ps:
            if p.ndim < 2:
                no_wd.append(p)
            else:
                with_wd.append(p)
        if with_wd:
            groups.append({"params": with_wd, "lr": base_lr * lr_mult,
                           "weight_decay": wd, "name": name})
        if no_wd:
            groups.append({"params": no_wd, "lr": base_lr * lr_mult,
                           "weight_decay": 0.0, "name": f"{name}_no_wd"})

    _add("range_enc", model.range_enc.parameters(), decay ** 4)
    _add("temp_stage0", list(model.temporal_enc.stage0.parameters()) +
         list(model.temporal_enc.ds0.parameters()), decay ** 3)
    _add("temp_stage1", list(model.temporal_enc.stage1.parameters()) +
         list(model.temporal_enc.ds1.parameters()), decay ** 2)
    _add("temp_stage2", model.temporal_enc.stage2.parameters(), decay ** 1)
    _add("temporal_attn", model.temporal_attn.parameters(), decay ** 0.5)
    _add("head", model.head.parameters(), 1.0)

    n_groups = len(groups)
    lrs = sorted(set(g["lr"] for g in groups))
    log.info("LLRD: %d groups, LR [%.2e … %.2e]", n_groups, min(lrs), max(lrs))
    return groups


# ===========================================================================
#  PSEUDO-LABELING from 2D CNN teachers (at 224×224)
# ===========================================================================


def generate_pseudo_labels(eval_keys, eval_cache, teacher_dirs, dev,
                           threshold=0.70):
    log.info("Generating pseudo-labels (threshold=%.2f) …", threshold)

    ds = RadarDataset(
        eval_keys, [0] * len(eval_keys), eval_cache, augment=False,
        img_size=CFG["teacher_img_size"],
    )
    loader = DataLoader(ds, batch_size=96, shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True)

    teachers = []
    for tdir in teacher_dirs:
        if not tdir.exists():
            continue
        for fold in range(CFG["n_folds"]):
            ckpt_path = tdir / f"best_fold{fold}.pt"
            if not ckpt_path.exists():
                continue
            m = ConvNeXtGeM(CFG["teacher_model_name"], CFG["n_classes"],
                            pretrained=False).to(dev)
            ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            teachers.append(m)

    n_t = len(teachers)
    log.info("  Loaded %d teacher models", n_t)

    all_probs = []
    with torch.no_grad():
        for x, _, _ in tqdm(loader, desc="  PL forward", leave=False):
            x = x.to(dev, non_blocking=True)
            bp = torch.zeros(x.size(0), CFG["n_classes"], device=dev)
            for m in teachers:
                bp += F.softmax(m(x), dim=1)
            bp /= n_t
            all_probs.append(bp.cpu())

    probs = torch.cat(all_probs)
    max_probs, pseudo_y = probs.max(dim=1)

    mask = max_probs >= threshold
    pl_keys = [eval_keys[i] for i in range(len(eval_keys)) if mask[i]]
    pl_labels = pseudo_y[mask].tolist()
    pl_soft = {eval_keys[i]: probs[i] for i in range(len(eval_keys)) if mask[i]}

    n_kept = len(pl_keys)
    log.info("  Pseudo-labeled: %d / %d kept (%.1f%%); avg conf = %.3f",
             n_kept, len(eval_keys), 100 * n_kept / len(eval_keys),
             max_probs[mask].mean().item() if n_kept > 0 else 0)

    del teachers
    torch.cuda.empty_cache()
    return pl_keys, pl_labels, pl_soft


# ===========================================================================
#  TEACHER SOFT LABELS for training data (KD at 224×224)
# ===========================================================================


def precompute_teacher_labels(keys, cache, teacher_dirs, dev):
    log.info("Pre-computing teacher soft labels (at 224×224) …")

    ds = RadarDataset(keys, [0] * len(keys), cache, augment=False,
                      img_size=CFG["teacher_img_size"])
    loader = DataLoader(ds, batch_size=96, shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True)

    teacher_models = []
    for tdir in teacher_dirs:
        if not tdir.exists():
            continue
        for fold in range(CFG["n_folds"]):
            ckpt_path = tdir / f"best_fold{fold}.pt"
            if not ckpt_path.exists():
                continue
            m = ConvNeXtGeM(CFG["teacher_model_name"], CFG["n_classes"],
                            pretrained=False).to(dev)
            ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            teacher_models.append(m)

    n_t = len(teacher_models)
    log.info("  Loaded %d teacher models", n_t)

    all_probs = []
    with torch.no_grad():
        for x, _, _ in tqdm(loader, desc="  Teacher fwd", leave=False):
            x = x.to(dev, non_blocking=True)
            bp = torch.zeros(x.size(0), CFG["n_classes"], device=dev)
            for m in teacher_models:
                bp += F.softmax(m(x), dim=1)
            bp /= n_t
            all_probs.append(bp.cpu())

    soft_all = torch.cat(all_probs)
    soft_dict = {k: soft_all[i] for i, k in enumerate(keys)}

    del teacher_models
    torch.cuda.empty_cache()
    log.info("  Soft labels ready: %d samples from %d teachers",
             len(soft_dict), n_t)
    return soft_dict


# ===========================================================================
#  MIXUP + CUTMIX
# ===========================================================================


def rand_bbox(H, W, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)
    return (
        max(0, cy - cut_h // 2), min(H, cy + cut_h // 2),
        max(0, cx - cut_w // 2), min(W, cx + cut_w // 2),
    )


def mix_or_cut(x, y, soft):
    idx = torch.randperm(x.size(0), device=x.device)
    use_cutmix = np.random.random() < CFG["cutmix_prob"]
    if use_cutmix:
        lam = float(np.random.beta(CFG["cutmix_alpha"], CFG["cutmix_alpha"]))
        t0, t1, r0, r1 = rand_bbox(x.size(2), x.size(3), lam)
        x_m = x.clone()
        x_m[:, :, t0:t1, r0:r1] = x[idx, :, t0:t1, r0:r1]
        lam = 1.0 - (t1 - t0) * (r1 - r0) / (x.size(2) * x.size(3))
    else:
        lam = float(np.random.beta(CFG["mixup_alpha"], CFG["mixup_alpha"]))
        x_m = lam * x + (1 - lam) * x[idx]
    return x_m, y, y[idx], soft, soft[idx], lam


# ===========================================================================
#  KD LOSS
# ===========================================================================


def kd_loss_fn(logits, soft_target, tau):
    log_pred = F.log_softmax(logits / tau, dim=1)
    soft_tau = (soft_target + 1e-8).pow(1.0 / tau)
    soft_tau = soft_tau / soft_tau.sum(dim=1, keepdim=True)
    return F.kl_div(log_pred, soft_tau, reduction="batchmean") * (tau ** 2)


# ===========================================================================
#  TRAIN / EVAL
# ===========================================================================


def train_epoch(model, loader, optim, sched, crit, dev, alpha, tau):
    model.train()
    tot_loss, correct, n = 0.0, 0.0, 0
    for x, y, soft in tqdm(loader, desc="  train", leave=False):
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        soft = soft.to(dev, non_blocking=True)

        x_m, ya, yb, sa, sb, lam = mix_or_cut(x, y, soft)
        logits = model(x_m)
        ce = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)
        kd = lam * kd_loss_fn(logits, sa, tau) + (1 - lam) * kd_loss_fn(logits, sb, tau)
        loss = alpha * ce + (1 - alpha) * kd

        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip_norm"])
        optim.step()
        sched.step()

        bs = x.size(0)
        tot_loss += loss.item() * bs
        preds = logits.argmax(1)
        correct += (
            lam * preds.eq(ya).float() + (1 - lam) * preds.eq(yb).float()
        ).sum().item()
        n += bs
    return tot_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, crit, dev):
    model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    for x, y, _ in tqdm(loader, desc="  val  ", leave=False):
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        logits = model(x)
        tot_loss += crit(logits, y).item() * x.size(0)
        correct += logits.argmax(1).eq(y).sum().item()
        n += x.size(0)
    return tot_loss / n, correct / n


# ===========================================================================
#  TTA (native resolution)
# ===========================================================================


def _tta_transforms(x):
    views = [x]
    views.append(x.flip(dims=[2]))
    views.append(x.roll(-4, dims=2))
    views.append(x.roll(4, dims=2))
    views.append(x[:, [1, 2, 0], :, :])
    views.append(x[:, [2, 0, 1], :, :])
    views.append(x + 0.05 * torch.randn_like(x))
    views.append(x + 0.08 * torch.randn_like(x))
    views.append(x * 0.9)
    views.append(x * 1.1)
    return views


@torch.no_grad()
def predict_ensemble_tta(fold_models, loader, dev, use_tta=True):
    for m in fold_models:
        m.eval()
    all_probs = []
    for x, _, _ in tqdm(loader, desc="  predict", leave=False):
        x = x.to(dev, non_blocking=True)
        bp = torch.zeros(x.size(0), CFG["n_classes"], device=dev)
        views = _tta_transforms(x) if use_tta else [x]
        for view in views:
            for model in fold_models:
                bp += F.softmax(model(view), dim=1)
        bp /= len(views) * len(fold_models)
        all_probs.append(bp.cpu())
    return torch.cat(all_probs)


def _write_submission(keys, preds, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "Pred"])
        for k, p in zip(keys, preds):
            sid = k.split("_", 1)[1]
            w.writerow([sid, int(p)])


# ===========================================================================
#  MAIN
# ===========================================================================


def _build_model(dev):
    return RangeTemporalNetV2(
        CFG["n_classes"],
        range_dim=CFG["range_dim"],
        temp_dims=tuple(CFG["temp_dims"]),
        temp_kernel=CFG["temp_kernel"],
        drop_path_rate=CFG["drop_path_rate"],
        drop_rate=CFG["drop_rate"],
        ms_samples=CFG["ms_dropout_samples"],
        attn_heads=CFG["attn_heads"],
        attn_layers=CFG["attn_layers"],
        attn_dropout=CFG["attn_dropout"],
    ).to(dev)


def main():
    out = Path(CFG["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.backends.cudnn.benchmark = True

    dev = torch.device("cuda:0")
    log.info("GPU : %s", torch.cuda.get_device_name(0))
    log.info("Config : %s", json.dumps(CFG, indent=2))

    train_samples, idx_to_cls = collect_train_samples(CFG["data_dir"])
    eval_samples = collect_eval_samples(CFG["data_dir"])
    keys = [s[1] for s in train_samples]
    labels = np.array([s[2] for s in train_samples])
    eval_keys = [s[1] for s in eval_samples]
    log.info("Train: %d  |  Eval: %d  |  Classes: %d",
             len(keys), len(eval_samples), len(idx_to_cls))

    log.info("Pre-loading training data …")
    train_cache = preload(train_samples)
    log.info("Pre-loading eval data …")
    eval_cache = preload(eval_samples, is_test=True)

    with open(out / "class_mapping.json", "w") as f:
        json.dump({str(k): v for k, v in idx_to_cls.items()}, f, indent=2)
    with open(out / "config.json", "w") as f:
        json.dump(CFG, f, indent=2)

    teacher_dirs = [V7_DIR, V6_DIR]

    pl_keys, pl_labels, pl_soft = generate_pseudo_labels(
        eval_keys, eval_cache, teacher_dirs, dev,
        threshold=CFG["pseudo_label_threshold"],
    )

    soft_labels = precompute_teacher_labels(keys, train_cache, teacher_dirs, dev)
    soft_labels.update(pl_soft)

    combined_cache = {**train_cache, **eval_cache}

    alpha = CFG["kd_alpha"]
    tau = CFG["kd_tau"]
    log.info("KD: α=%.2f  τ=%.1f  | PL: %d extra samples",
             alpha, tau, len(pl_keys))

    # ---- Model size check ------------------------------------------------
    probe = _build_model(torch.device("cpu"))
    n_params = sum(p.numel() for p in probe.parameters())
    log.info("RangeTemporalNetV2: %.2fM params (v1 CNN + temporal attention)",
             n_params / 1e6)
    del probe

    # ---- 5-fold CV --------------------------------------------------------
    skf = StratifiedKFold(
        n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"]
    )
    fold_accs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(keys, labels)):
        log.info("=" * 64)
        log.info("FOLD %d / %d", fold + 1, CFG["n_folds"])
        log.info("=" * 64)

        tr_keys = [keys[i] for i in tr_idx] + pl_keys
        tr_labels = labels[tr_idx].tolist() + pl_labels
        va_keys = [keys[i] for i in va_idx]
        va_labels = labels[va_idx].tolist()

        log.info("  Train: %d (orig %d + PL %d)  |  Val: %d",
                 len(tr_keys), len(tr_idx), len(pl_keys), len(va_keys))

        tr_ds = RadarDataset(tr_keys, tr_labels, combined_cache, augment=True,
                             soft_labels=soft_labels, img_size=None)
        va_ds = RadarDataset(va_keys, va_labels, combined_cache, augment=False,
                             img_size=None)

        tr_loader = DataLoader(
            tr_ds, batch_size=CFG["batch_size"], shuffle=True,
            num_workers=CFG["num_workers"], pin_memory=True,
            drop_last=True, persistent_workers=True,
        )
        va_loader = DataLoader(
            va_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
            num_workers=CFG["num_workers"], pin_memory=True,
            persistent_workers=True,
        )

        model = _build_model(dev)

        # Born-again from v1
        v1_ckpt = V1_DIR / f"best_fold{fold}.pt"
        if v1_ckpt.exists():
            load_born_again(model, v1_ckpt, dev)
        else:
            log.info("  No v1 checkpoint for fold %d — training from scratch", fold)

        log.info("Model: RangeTemporalNetV2  |  %.2fM params", n_params / 1e6)

        crit = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
        best_acc = 0.0
        ckpt_path = out / f"best_fold{fold}.pt"

        # ==== STAGE 1: Head-only + attention warmup ========================
        log.info("── Stage 1: attention + head warmup (%d epochs) ──",
                 CFG["stage1_epochs"])

        for p in model.range_enc.parameters():
            p.requires_grad = False
        for p in model.temporal_enc.parameters():
            p.requires_grad = False

        s1_params = [p for p in model.parameters() if p.requires_grad]
        opt = AdamW(s1_params, lr=CFG["stage1_lr"],
                    weight_decay=CFG["weight_decay"])
        steps = len(tr_loader) * CFG["stage1_epochs"]
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=CFG["stage1_lr"], total_steps=steps,
            pct_start=0.3, anneal_strategy="cos",
            div_factor=10.0, final_div_factor=100.0,
        )

        for epoch in range(1, CFG["stage1_epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(
                model, tr_loader, opt, sched, crit, dev, alpha, tau,
            )
            va_loss, va_acc = eval_epoch(model, va_loader, crit, dev)
            dt = time.time() - t0
            log.info(
                "  S1 Ep %2d/%d  train %.4f / %.1f%%  val %.4f / %.1f%%    %.0fs",
                epoch, CFG["stage1_epochs"], tr_loss, 100 * tr_acc,
                va_loss, 100 * va_acc, dt,
            )
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({
                    "fold": fold, "epoch": epoch, "stage": 1,
                    "model_state_dict": model.state_dict(),
                    "val_acc": va_acc,
                }, ckpt_path)
                log.info("  ★ best %.2f%%", 100 * va_acc)

        log.info("Stage 1 done — best val: %.2f%%", 100 * best_acc)
        del opt, sched

        # ==== STAGE 2: Full fine-tuning with LLRD ==========================
        log.info("── Stage 2: full fine-tuning + LLRD (%d epochs, patience %d) ──",
                 CFG["stage2_epochs"], CFG["early_stopping_patience"])

        for p in model.parameters():
            p.requires_grad = True

        param_groups = get_llrd_params(
            model, CFG["stage2_lr"],
            decay=CFG["llrd_decay"], wd=CFG["weight_decay"],
        )
        opt = AdamW(param_groups, lr=CFG["stage2_lr"],
                    weight_decay=CFG["weight_decay"])
        steps = len(tr_loader) * CFG["stage2_epochs"]
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in param_groups],
            total_steps=steps, pct_start=CFG["warmup_pct"],
            anneal_strategy="cos",
            div_factor=25.0, final_div_factor=1000.0,
        )

        patience = 0
        for epoch in range(1, CFG["stage2_epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(
                model, tr_loader, opt, sched, crit, dev, alpha, tau,
            )
            va_loss, va_acc = eval_epoch(model, va_loader, crit, dev)
            dt = time.time() - t0
            lr_now = opt.param_groups[-1]["lr"]
            log.info(
                "  S2 Ep %3d/%d  train %.4f / %.1f%%  val %.4f / %.1f%%  "
                "lr %.2e  %4.0fs",
                epoch, CFG["stage2_epochs"], tr_loss, 100 * tr_acc,
                va_loss, 100 * va_acc, lr_now, dt,
            )
            if va_acc > best_acc:
                best_acc = va_acc
                patience = 0
                torch.save({
                    "fold": fold, "epoch": epoch, "stage": 2,
                    "model_state_dict": model.state_dict(),
                    "val_acc": va_acc,
                }, ckpt_path)
                log.info("  ★ best %.2f%% → %s", 100 * va_acc, ckpt_path)
            else:
                patience += 1
                if patience >= CFG["early_stopping_patience"]:
                    log.info("  Early stopping at S2 epoch %d", epoch)
                    break

        fold_accs.append(best_acc)
        log.info("  Fold %d best val acc: %.2f%%", fold + 1, 100 * best_acc)
        del model, opt, sched, tr_loader, va_loader
        torch.cuda.empty_cache()

    # ---- CV summary -------------------------------------------------------
    mean_a, std_a = np.mean(fold_accs), np.std(fold_accs)
    log.info("=" * 64)
    log.info("5-FOLD CV RESULTS")
    log.info("=" * 64)
    for i, a in enumerate(fold_accs):
        log.info("  Fold %d : %.2f%%", i + 1, 100 * a)
    log.info("  Mean  : %.2f%% ± %.2f%%", 100 * mean_a, 100 * std_a)

    with open(out / "cv_results.json", "w") as f:
        json.dump({
            "fold_accuracies": [float(a) for a in fold_accs],
            "mean": float(mean_a), "std": float(std_a),
        }, f, indent=2)

    # ---- Submission -------------------------------------------------------
    eval_ds = RadarDataset(eval_keys, [0] * len(eval_keys), eval_cache,
                           augment=False, img_size=None)
    eval_loader = DataLoader(
        eval_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )

    log.info("Loading v2 checkpoints for ensemble …")
    models_v2 = []
    for fold in range(CFG["n_folds"]):
        m = RangeTemporalNetV2(
            CFG["n_classes"],
            range_dim=CFG["range_dim"],
            temp_dims=tuple(CFG["temp_dims"]),
            temp_kernel=CFG["temp_kernel"],
            drop_path_rate=0.0,
            drop_rate=CFG["drop_rate"],
            ms_samples=CFG["ms_dropout_samples"],
            attn_heads=CFG["attn_heads"],
            attn_layers=CFG["attn_layers"],
            attn_dropout=0.0,
        ).to(dev)
        ckpt = torch.load(out / f"best_fold{fold}.pt", map_location=dev,
                          weights_only=True)
        m.load_state_dict(ckpt["model_state_dict"])
        models_v2.append(m)

    probs_no = predict_ensemble_tta(models_v2, eval_loader, dev, use_tta=False)
    preds_no = probs_no.argmax(dim=1).numpy()
    _write_submission(eval_keys, preds_no, out / "submission_no_tta.csv")
    log.info("Submission (no TTA) → %s  (%d rows)",
             out / "submission_no_tta.csv", len(preds_no))

    log.info("Running 10-way TTA × 5 folds …")
    probs_v2 = predict_ensemble_tta(models_v2, eval_loader, dev, use_tta=True)
    preds_v2 = probs_v2.argmax(dim=1).numpy()
    _write_submission(eval_keys, preds_v2, out / "submission_tta.csv")
    log.info("Submission (TTA) → %s", out / "submission_tta.csv")
    changed = (preds_v2 != preds_no).sum()
    log.info("TTA changed %d / %d (%.1f%%)", changed, len(preds_v2),
             100 * changed / len(preds_v2))

    torch.save({"keys": eval_keys, "probs": probs_v2},
               out / "eval_probs_tta.pt")
    log.info("Saved v2 TTA probs → %s", out / "eval_probs_tta.pt")
    del models_v2

    # ---- Cross-arch ensemble: 1Dv2 + CNN v7 ------------------------------
    if V7_DIR.exists() and (V7_DIR / "best_fold0.pt").exists():
        log.info("Loading CNN v7 for cross-arch ensemble …")
        eval_ds_224 = RadarDataset(
            eval_keys, [0] * len(eval_keys), eval_cache, augment=False,
            img_size=CFG["teacher_img_size"],
        )
        eval_loader_224 = DataLoader(
            eval_ds_224, batch_size=64, shuffle=False,
            num_workers=CFG["num_workers"], pin_memory=True,
        )

        cnn_models = []
        for fold in range(CFG["n_folds"]):
            m = ConvNeXtGeM(CFG["teacher_model_name"], CFG["n_classes"],
                            pretrained=False).to(dev)
            ckpt = torch.load(V7_DIR / f"best_fold{fold}.pt", map_location=dev,
                              weights_only=True)
            m.load_state_dict(ckpt["model_state_dict"])
            cnn_models.append(m)

        probs_cnn = predict_ensemble_tta(cnn_models, eval_loader_224, dev,
                                         use_tta=True)
        del cnn_models
        torch.cuda.empty_cache()

        for w_1d in [0.3, 0.4, 0.5]:
            w_cnn = 1.0 - w_1d
            probs_ens = w_1d * probs_v2 + w_cnn * probs_cnn
            preds_ens = probs_ens.argmax(dim=1).numpy()
            tag = f"1dv2_{int(w_1d*100)}_cnn{int(w_cnn*100)}"
            _write_submission(eval_keys, preds_ens,
                              out / f"submission_{tag}_ensemble.csv")
            diff = (preds_ens != preds_v2).sum()
            log.info("  %s ensemble: %d changed from 1Dv2-only", tag, diff)

    log.info("Done.")


if __name__ == "__main__":
    main()
