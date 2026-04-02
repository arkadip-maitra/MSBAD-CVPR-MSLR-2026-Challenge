#!/usr/bin/env python3
"""
CVPR MSLR 2026 Track 2 — Italian Sign Language Recognition (Radar)
v3: ResNet-18 + CBAM + EMA + dual pooling + discriminative LR.

Improvements over v2 (82.4% CV):
  - Exponential Moving Average (EMA, decay=0.9998) of weights for eval
  - Dual pooling: GAP + GMP → 1024-dim (richer features)
  - Discriminative learning rates (lower LR for earlier layers)
  - Amplitude scaling augmentation
  - Wider time warp range (0.7–1.3)
  - Random circular time shift
  - Random erasing (p=0.25)
  - 10-way TTA
  - 150 epochs, patience 30

Usage:
    conda run -n slr python /data/slr/train_v3.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomErasing
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
OUT_DIR = Path("/data/slr/checkpoints_v3")
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CFG = {
    "data_dir": "/data/slr/track-2",
    "output_dir": str(OUT_DIR),
    "max_frames": 48,
    "n_range_bins": 256,
    "n_classes": 126,
    "channel_means": [-64.3217, -62.9119, -63.9254],
    "channel_stds": [12.5721, 11.6620, 11.7259],
    # Training
    "n_folds": 5,
    "batch_size": 64,
    "num_workers": 4,
    "epochs": 150,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "label_smoothing": 0.1,
    "early_stopping_patience": 30,
    "grad_clip_norm": 1.0,
    # Discriminative LR: multiplier for each ResNet stage
    "lr_mult_conv1": 0.1,
    "lr_mult_layer1": 0.2,
    "lr_mult_layer2": 0.4,
    "lr_mult_layer3": 0.7,
    "lr_mult_layer4": 1.0,
    "lr_mult_head": 1.0,
    # EMA
    "ema_decay": 0.9998,
    # Augmentation
    "mixup_alpha": 0.4,
    "cutmix_alpha": 1.0,
    "cutmix_prob": 0.5,
    "time_mask_max": 8,
    "range_mask_max": 30,
    "noise_std": 0.1,
    "time_warp_lo": 0.7,
    "time_warp_hi": 1.3,
    "time_reverse_prob": 0.5,
    "channel_shuffle_prob": 0.2,
    "channel_drop_prob": 0.1,
    "amp_scale_lo": 0.85,
    "amp_scale_hi": 1.15,
    "time_shift_max": 6,
    "random_erase_prob": 0.25,
    # Model
    "pretrained": True,
    "dropout": 0.3,
    "seed": 42,
}


# ===========================================================================
#  DATA
# ===========================================================================


def collect_train_samples(data_dir: str):
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


def collect_test_samples(data_dir: str):
    val_dir = os.path.join(data_dir, "val")
    samples = []
    for sname in sorted(os.listdir(val_dir)):
        spath = os.path.join(val_dir, sname)
        if os.path.isdir(spath):
            samples.append((spath, sname))
    return samples


def _load_one(sample_path: str, sample_name: str) -> np.ndarray:
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
    def __init__(self, keys, labels, cache, augment=False):
        self.keys = keys
        self.labels = labels
        self.cache = cache
        self.augment = augment

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
            pad = np.zeros((3, max_t - t, x.shape[2]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        return x

    def _apply_aug(self, x):
        max_t = CFG["max_frames"]

        # Time reversal
        if np.random.random() < CFG["time_reverse_prob"]:
            x = x[:, ::-1, :].copy()

        # Channel shuffle
        if np.random.random() < CFG["channel_shuffle_prob"]:
            x = x[np.random.permutation(3)]

        # Channel dropout
        if np.random.random() < CFG["channel_drop_prob"]:
            x[np.random.randint(0, 3)] = 0.0

        # Random circular time shift
        if np.random.random() < 0.3:
            shift = np.random.randint(-CFG["time_shift_max"],
                                      CFG["time_shift_max"] + 1)
            x = np.roll(x, shift, axis=1)

        # Double time masking
        for _ in range(2):
            if np.random.random() < 0.5:
                w = np.random.randint(1, CFG["time_mask_max"] + 1)
                t0 = np.random.randint(0, max(1, x.shape[1] - w))
                x[:, t0 : t0 + w, :] = 0.0

        # Double range masking
        for _ in range(2):
            if np.random.random() < 0.5:
                w = np.random.randint(1, CFG["range_mask_max"] + 1)
                r0 = np.random.randint(0, max(1, x.shape[2] - w))
                x[:, :, r0 : r0 + w] = 0.0

        # Amplitude scaling (per-sample random gain)
        if np.random.random() < 0.5:
            gain = np.random.uniform(CFG["amp_scale_lo"], CFG["amp_scale_hi"])
            x = x * gain

        # Gaussian noise
        if np.random.random() < 0.5:
            x = x + np.random.normal(0, CFG["noise_std"], x.shape).astype(
                np.float32
            )

        # Time warp
        if np.random.random() < 0.3:
            factor = np.random.uniform(CFG["time_warp_lo"], CFG["time_warp_hi"])
            new_t = max(1, int(x.shape[1] * factor))
            x = zoom(x, (1, new_t / x.shape[1], 1), order=1).astype(np.float32)
            x = self._pad_or_crop(x, max_t, random_crop=True)

        return x

    def __getitem__(self, idx):
        x = self.cache[self.keys[idx]].copy()
        x = (x - _MEANS) / _STDS
        x = self._pad_or_crop(x, CFG["max_frames"], random_crop=self.augment)
        if self.augment:
            x = self._apply_aug(x)
        t = torch.from_numpy(x)
        if self.augment:
            t = _ERASER(t)
        return t, self.labels[idx]


# ===========================================================================
#  CBAM
# ===========================================================================


class ChannelAttn(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        mid = max(c // r, 1)
        self.mlp = nn.Sequential(
            nn.Linear(c, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c, bias=False),
        )

    def forward(self, x):
        avg = self.mlp(x.mean(dim=[2, 3]))
        mx = self.mlp(x.amax(dim=[2, 3]))
        return x * torch.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)


class SpatialAttn(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        desc = torch.cat(
            [x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1
        )
        return x * torch.sigmoid(self.bn(self.conv(desc)))


class CBAM(nn.Module):
    def __init__(self, c, r=16, k=7):
        super().__init__()
        self.ca = ChannelAttn(c, r)
        self.sa = SpatialAttn(k)

    def forward(self, x):
        return self.sa(self.ca(x))


# ===========================================================================
#  ResNet-18 + CBAM + Dual Pooling
# ===========================================================================


class ResNet18CBAM_v3(nn.Module):
    """
    Same backbone as v1/v2 but with dual pooling (GAP + GMP → 1024-dim).

    Feature-map progression for (3, 48, 256) input:
      conv1  → (64,  48, 128)   stride (1,2), no maxpool
      layer1 → (64,  48, 128) + CBAM
      layer2 → (128, 24,  64) + CBAM
      layer3 → (256, 12,  32) + CBAM
      layer4 → (512,  6,  16) + CBAM
      GAP+GMP → (1024,)
    """

    def __init__(self, n_cls=126, pretrained=True, dropout=0.3):
        super().__init__()
        w = "IMAGENET1K_V1" if pretrained else None
        base = models.resnet18(weights=w)

        self.conv1 = nn.Conv2d(3, 64, 7, stride=(1, 2), padding=3, bias=False)
        if pretrained:
            self.conv1.weight.data.copy_(base.conv1.weight.data)
        self.bn1 = base.bn1
        self.relu = base.relu

        self.layer1 = base.layer1
        self.cbam1 = CBAM(64)
        self.layer2 = base.layer2
        self.cbam2 = CBAM(128)
        self.layer3 = base.layer3
        self.cbam3 = CBAM(256)
        self.layer4 = base.layer4
        self.cbam4 = CBAM(512)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, n_cls)  # 512 GAP + 512 GMP

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        avg = self.gap(x).flatten(1)
        mx = self.gmp(x).flatten(1)
        return self.fc(self.drop(torch.cat([avg, mx], dim=1)))


# ===========================================================================
#  EMA (Exponential Moving Average)
# ===========================================================================


class ModelEMA:
    """Maintains an EMA shadow copy of model weights for evaluation."""

    def __init__(self, model, decay=0.9998):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for sp, mp in zip(self.shadow.parameters(), model.parameters()):
            sp.data.mul_(self.decay).add_(mp.data, alpha=1.0 - self.decay)
        for sb, mb in zip(self.shadow.buffers(), model.buffers()):
            sb.data.copy_(mb.data)

    def state_dict(self):
        return self.shadow.state_dict()

    def eval(self):
        self.shadow.eval()
        return self.shadow


# ===========================================================================
#  DISCRIMINATIVE LEARNING RATES
# ===========================================================================


def get_param_groups(model, base_lr, wd):
    """Group parameters with layer-specific learning rates."""
    groups = {
        "conv1_bn1": {"params": [], "lr": base_lr * CFG["lr_mult_conv1"]},
        "layer1":    {"params": [], "lr": base_lr * CFG["lr_mult_layer1"]},
        "layer2":    {"params": [], "lr": base_lr * CFG["lr_mult_layer2"]},
        "layer3":    {"params": [], "lr": base_lr * CFG["lr_mult_layer3"]},
        "layer4":    {"params": [], "lr": base_lr * CFG["lr_mult_layer4"]},
        "head":      {"params": [], "lr": base_lr * CFG["lr_mult_head"]},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("conv1", "bn1")):
            groups["conv1_bn1"]["params"].append(param)
        elif name.startswith("layer1") or name.startswith("cbam1"):
            groups["layer1"]["params"].append(param)
        elif name.startswith("layer2") or name.startswith("cbam2"):
            groups["layer2"]["params"].append(param)
        elif name.startswith("layer3") or name.startswith("cbam3"):
            groups["layer3"]["params"].append(param)
        elif name.startswith("layer4") or name.startswith("cbam4"):
            groups["layer4"]["params"].append(param)
        else:
            groups["head"]["params"].append(param)

    result = []
    for gname, g in groups.items():
        if g["params"]:
            no_decay = [p for p in g["params"] if p.ndim <= 1]
            decay = [p for p in g["params"] if p.ndim > 1]
            if decay:
                result.append({"params": decay, "lr": g["lr"],
                               "weight_decay": wd})
            if no_decay:
                result.append({"params": no_decay, "lr": g["lr"],
                               "weight_decay": 0.0})

    log.info("Discriminative LR groups:")
    for gname, g in groups.items():
        n = sum(p.numel() for p in g["params"])
        log.info("  %-10s  lr=%.2e  params=%d", gname, g["lr"], n)
    return result


# ===========================================================================
#  MIXUP + CUTMIX
# ===========================================================================


def rand_bbox(H, W, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)
    t0 = max(0, cy - cut_h // 2)
    t1 = min(H, cy + cut_h // 2)
    r0 = max(0, cx - cut_w // 2)
    r1 = min(W, cx + cut_w // 2)
    return t0, t1, r0, r1


def mix_or_cut(x, y, mixup_alpha, cutmix_alpha, cutmix_prob):
    use_cutmix = np.random.random() < cutmix_prob
    if use_cutmix:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        t0, t1, r0, r1 = rand_bbox(x.size(2), x.size(3), lam)
        x_m = x.clone()
        x_m[:, :, t0:t1, r0:r1] = x[idx, :, t0:t1, r0:r1]
        lam = 1.0 - (t1 - t0) * (r1 - r0) / (x.size(2) * x.size(3))
        return x_m, y, y[idx], lam
    else:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ===========================================================================
#  TRAIN / EVAL
# ===========================================================================


def train_epoch(model, ema, loader, optim, sched, crit, dev):
    model.train()
    tot_loss, correct, n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)

        mx, ya, yb, lam = mix_or_cut(
            x, y, CFG["mixup_alpha"], CFG["cutmix_alpha"], CFG["cutmix_prob"]
        )
        logits = model(mx)
        loss = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip_norm"])
        optim.step()
        sched.step()
        ema.update(model)

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
    for x, y in tqdm(loader, desc="  val  ", leave=False):
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        logits = model(x)
        tot_loss += crit(logits, y).item() * x.size(0)
        correct += logits.argmax(1).eq(y).sum().item()
        n += x.size(0)
    return tot_loss / n, correct / n


# ===========================================================================
#  10-WAY TTA
# ===========================================================================


def _tta_transforms(x: torch.Tensor) -> list[torch.Tensor]:
    """10 augmented views for TTA on (B, 3, 48, 256) input."""
    views = [x]
    views.append(x.flip(dims=[2]))                     # 1: time-reversed
    views.append(x.roll(-3, dims=2))                   # 2: shift left 3
    views.append(x.roll(3, dims=2))                    # 3: shift right 3
    views.append(x[:, [1, 2, 0], :, :])                # 4: channel perm A
    views.append(x[:, [2, 0, 1], :, :])                # 5: channel perm B
    views.append(x + 0.05 * torch.randn_like(x))       # 6: noise A
    views.append(x + 0.08 * torch.randn_like(x))       # 7: noise B
    views.append(x * 0.9)                              # 8: amplitude down
    views.append(x * 1.1)                              # 9: amplitude up
    return views


@torch.no_grad()
def predict_ensemble_tta(fold_models, loader, dev, use_tta=True):
    for m in fold_models:
        m.eval()
    all_probs = []
    for x, _ in tqdm(loader, desc="  predict", leave=False):
        x = x.to(dev, non_blocking=True)
        batch_probs = torch.zeros(x.size(0), CFG["n_classes"], device=dev)
        views = _tta_transforms(x) if use_tta else [x]
        for view in views:
            for model in fold_models:
                batch_probs += F.softmax(model(view), dim=1)
        batch_probs /= len(views) * len(fold_models)
        all_probs.append(batch_probs.cpu())
    return torch.cat(all_probs)


# ===========================================================================
#  MAIN
# ===========================================================================


def main():
    out = Path(CFG["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.backends.cudnn.benchmark = True

    dev = torch.device("cuda:0")
    log.info("GPU : %s", torch.cuda.get_device_name(0))
    log.info("Config : %s", json.dumps(CFG, indent=2))

    # ---- collect & preload ------------------------------------------------
    train_samples, idx_to_cls = collect_train_samples(CFG["data_dir"])
    test_samples = collect_test_samples(CFG["data_dir"])

    keys = [s[1] for s in train_samples]
    labels = np.array([s[2] for s in train_samples])
    log.info(
        "Train: %d  |  Test: %d  |  Classes: %d",
        len(keys), len(test_samples), len(idx_to_cls),
    )

    log.info("Pre-loading training data …")
    train_cache = preload(train_samples)
    log.info("Pre-loading test data …")
    test_cache = preload(test_samples, is_test=True)

    with open(out / "class_mapping.json", "w") as f:
        json.dump({str(k): v for k, v in idx_to_cls.items()}, f, indent=2)
    with open(out / "config.json", "w") as f:
        json.dump(CFG, f, indent=2)

    # ---- 5-fold CV --------------------------------------------------------
    skf = StratifiedKFold(
        n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"]
    )
    fold_accs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(keys, labels)):
        log.info("=" * 64)
        log.info("FOLD %d / %d", fold + 1, CFG["n_folds"])
        log.info("=" * 64)

        tr_keys = [keys[i] for i in tr_idx]
        tr_labels = labels[tr_idx].tolist()
        va_keys = [keys[i] for i in va_idx]
        va_labels = labels[va_idx].tolist()

        tr_ds = RadarDataset(tr_keys, tr_labels, train_cache, augment=True)
        va_ds = RadarDataset(va_keys, va_labels, train_cache, augment=False)

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

        # ---- model + EMA --------------------------------------------------
        model = ResNet18CBAM_v3(
            CFG["n_classes"], CFG["pretrained"], CFG["dropout"]
        ).to(dev)
        ema = ModelEMA(model, decay=CFG["ema_decay"])

        n_params = sum(p.numel() for p in model.parameters())
        log.info("Model params: %.2fM", n_params / 1e6)

        # ---- optimizer with discriminative LR -----------------------------
        param_groups = get_param_groups(model, CFG["lr"], CFG["weight_decay"])
        optimizer = AdamW(param_groups)

        total_steps = len(tr_loader) * CFG["epochs"]
        warmup_frac = CFG["warmup_epochs"] / CFG["epochs"]
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=total_steps,
            pct_start=warmup_frac,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        crit = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

        best_acc, patience = 0.0, 0

        for epoch in range(1, CFG["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(
                model, ema, tr_loader, optimizer, sched, crit, dev
            )
            # Evaluate using EMA weights
            ema_model = ema.eval()
            va_loss, va_acc = eval_epoch(ema_model, va_loader, crit, dev)
            dt = time.time() - t0
            lr_now = optimizer.param_groups[-1]["lr"]

            log.info(
                "  Ep %3d/%d  train %.4f / %.1f%%  val(ema) %.4f / %.1f%%  "
                "lr %.2e  %4.0fs",
                epoch, CFG["epochs"], tr_loss, 100 * tr_acc,
                va_loss, 100 * va_acc, lr_now, dt,
            )

            if va_acc > best_acc:
                best_acc = va_acc
                patience = 0
                ckpt_path = out / f"best_fold{fold}.pt"
                torch.save({
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": ema.state_dict(),
                    "val_acc": va_acc,
                    "val_loss": va_loss,
                }, ckpt_path)
                log.info("  ★ best %.2f%% → %s", 100 * va_acc, ckpt_path)
            else:
                patience += 1
                if patience >= CFG["early_stopping_patience"]:
                    log.info("  Early stopping at epoch %d", epoch)
                    break

        fold_accs.append(best_acc)
        log.info("  Fold %d best val acc: %.2f%%", fold + 1, 100 * best_acc)

        del model, ema, optimizer, sched, tr_loader, va_loader
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
            "mean": float(mean_a),
            "std": float(std_a),
        }, f, indent=2)

    # ---- Ensemble test-set predictions with TTA ---------------------------
    log.info("Loading EMA checkpoints for ensemble + TTA …")
    test_keys = [s[1] for s in test_samples]
    test_ds = RadarDataset(
        test_keys, [0] * len(test_keys), test_cache, augment=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )

    fold_models = []
    for fold in range(CFG["n_folds"]):
        m = ResNet18CBAM_v3(
            CFG["n_classes"], pretrained=False, dropout=0.0
        ).to(dev)
        ckpt = torch.load(
            out / f"best_fold{fold}.pt", map_location=dev, weights_only=True
        )
        m.load_state_dict(ckpt["model_state_dict"])
        fold_models.append(m)

    # Without TTA
    probs_no = predict_ensemble_tta(fold_models, test_loader, dev, use_tta=False)
    preds_no = probs_no.argmax(dim=1).numpy()
    sub_path = out / "submission_no_tta.csv"
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Predicted"])
        for k, p in zip(test_keys, preds_no):
            w.writerow([k, int(p)])
    log.info("Submission (no TTA) → %s", sub_path)

    # With 10-way TTA
    log.info(
        "Running 10-way TTA × 5 folds = 50 forward passes per sample …"
    )
    probs_tta = predict_ensemble_tta(fold_models, test_loader, dev, use_tta=True)
    preds_tta = probs_tta.argmax(dim=1).numpy()
    sub_path = out / "submission_tta.csv"
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Predicted"])
        for k, p in zip(test_keys, preds_tta):
            w.writerow([k, int(p)])
    log.info("Submission (TTA)    → %s", sub_path)

    changed = (preds_tta != preds_no).sum()
    log.info(
        "TTA changed %d / %d predictions (%.1f%%)",
        changed, len(preds_tta), 100 * changed / len(preds_tta),
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
