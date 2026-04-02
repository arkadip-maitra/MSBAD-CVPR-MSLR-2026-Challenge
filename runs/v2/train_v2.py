#!/usr/bin/env python3
"""
CVPR MSLR 2026 Track 2 — Italian Sign Language Recognition (Radar)
v2: ResNet-18 + CBAM with CutMix, stronger augmentation, and TTA.

Improvements over v1 (78.5% CV):
  - CutMix alternating with Mixup
  - Time reversal augmentation
  - Channel shuffle augmentation
  - Double time/range masking
  - 6-way Test-Time Augmentation on inference
  - Increased training epochs (120) and patience (25)

Usage:
    conda run -n slr python /data/slr/train_v2.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
OUT_DIR = Path("/data/slr/checkpoints_v2")
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
    # Data shape
    "max_frames": 48,
    "n_range_bins": 256,
    "n_classes": 126,
    # Global z-score stats
    "channel_means": [-64.3217, -62.9119, -63.9254],
    "channel_stds": [12.5721, 11.6620, 11.7259],
    # Training
    "n_folds": 5,
    "batch_size": 64,
    "num_workers": 4,
    "epochs": 120,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "label_smoothing": 0.1,
    "early_stopping_patience": 25,
    "grad_clip_norm": 1.0,
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
    # TTA
    "tta_enabled": True,
    # Model
    "pretrained": True,
    "dropout": 0.3,
    # Misc
    "seed": 42,
}

# ===========================================================================
#  DATA COLLECTION & PRE-LOADING
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
            s = np.random.randint(0, t - max_t + 1) if random_crop else (t - max_t) // 2
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

        # Channel shuffle (permute RTM order)
        if np.random.random() < CFG["channel_shuffle_prob"]:
            perm = np.random.permutation(3)
            x = x[perm]

        # Channel dropout (zero out one RTM channel entirely)
        if np.random.random() < CFG["channel_drop_prob"]:
            ch = np.random.randint(0, 3)
            x[ch] = 0.0

        # Time masking (apply up to 2 masks)
        for _ in range(2):
            if np.random.random() < 0.5:
                w = np.random.randint(1, CFG["time_mask_max"] + 1)
                t0 = np.random.randint(0, max(1, x.shape[1] - w))
                x[:, t0 : t0 + w, :] = 0.0

        # Range masking (apply up to 2 masks)
        for _ in range(2):
            if np.random.random() < 0.5:
                w = np.random.randint(1, CFG["range_mask_max"] + 1)
                r0 = np.random.randint(0, max(1, x.shape[2] - w))
                x[:, :, r0 : r0 + w] = 0.0

        # Gaussian noise
        if np.random.random() < 0.5:
            x = x + np.random.normal(0, CFG["noise_std"], x.shape).astype(np.float32)

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
        return torch.from_numpy(x), self.labels[idx]


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
        desc = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.bn(self.conv(desc)))


class CBAM(nn.Module):
    def __init__(self, c, r=16, k=7):
        super().__init__()
        self.ca = ChannelAttn(c, r)
        self.sa = SpatialAttn(k)

    def forward(self, x):
        return self.sa(self.ca(x))


# ===========================================================================
#  ResNet-18 + CBAM
# ===========================================================================


class ResNet18CBAM(nn.Module):
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

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(512, n_cls)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        return self.fc(self.drop(self.pool(x).flatten(1)))


# ===========================================================================
#  MIXUP + CUTMIX
# ===========================================================================


def rand_bbox(H, W, lam):
    """Return (t0, t1, r0, r1) for a CutMix box whose area ratio ≈ 1-lam."""
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    t0 = max(0, cy - cut_h // 2)
    t1 = min(H, cy + cut_h // 2)
    r0 = max(0, cx - cut_w // 2)
    r1 = min(W, cx + cut_w // 2)
    return t0, t1, r0, r1


def mix_or_cut(x, y, mixup_alpha, cutmix_alpha, cutmix_prob):
    """Randomly apply either Mixup or CutMix to a batch."""
    use_cutmix = np.random.random() < cutmix_prob

    if use_cutmix:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        t0, t1, r0, r1 = rand_bbox(x.size(2), x.size(3), lam)
        x_mixed = x.clone()
        x_mixed[:, :, t0:t1, r0:r1] = x[idx, :, t0:t1, r0:r1]
        lam = 1.0 - (t1 - t0) * (r1 - r0) / (x.size(2) * x.size(3))
        return x_mixed, y, y[idx], lam
    else:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ===========================================================================
#  TRAIN / EVAL LOOPS
# ===========================================================================


def train_epoch(model, loader, optim, sched, crit, dev):
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
#  TEST-TIME AUGMENTATION
# ===========================================================================


def _tta_transforms(x: torch.Tensor) -> list[torch.Tensor]:
    """Generate 6 augmented versions of a batch for TTA.

    Args:
        x: (B, 3, T, 256) tensor

    Returns:
        List of 6 (B, 3, T, 256) tensors:
          0 - original
          1 - time-reversed
          2 - shifted left 2 frames (circular)
          3 - shifted right 2 frames (circular)
          4 - channel-permuted [1, 2, 0]
          5 - slight Gaussian noise
    """
    views = [x]
    views.append(x.flip(dims=[2]))
    views.append(x.roll(-2, dims=2))
    views.append(x.roll(2, dims=2))
    views.append(x[:, [1, 2, 0], :, :])
    views.append(x + 0.05 * torch.randn_like(x))
    return views


@torch.no_grad()
def predict_ensemble_tta(fold_models, loader, dev, use_tta=True):
    """Ensemble prediction with optional TTA."""
    for m in fold_models:
        m.eval()
    all_probs = []

    for x, _ in tqdm(loader, desc="  predict", leave=False):
        x = x.to(dev, non_blocking=True)
        batch_probs = torch.zeros(x.size(0), CFG["n_classes"], device=dev)

        if use_tta:
            views = _tta_transforms(x)
        else:
            views = [x]

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

        model = ResNet18CBAM(
            CFG["n_classes"], CFG["pretrained"], CFG["dropout"]
        ).to(dev)

        optimizer = AdamW(
            model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
        )

        total_steps = len(tr_loader) * CFG["epochs"]
        warmup_frac = CFG["warmup_epochs"] / CFG["epochs"]
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CFG["lr"], total_steps=total_steps,
            pct_start=warmup_frac, anneal_strategy="cos",
            div_factor=25.0, final_div_factor=1000.0,
        )

        crit = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

        best_acc, patience = 0.0, 0

        for epoch in range(1, CFG["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(
                model, tr_loader, optimizer, sched, crit, dev
            )
            va_loss, va_acc = eval_epoch(model, va_loader, crit, dev)
            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]

            log.info(
                "  Ep %3d/%d  train %.4f / %.1f%%  val %.4f / %.1f%%  "
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
                    "model_state_dict": model.state_dict(),
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

        del model, optimizer, sched, tr_loader, va_loader
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
    log.info("Loading best checkpoints for ensemble + TTA …")
    test_keys = [s[1] for s in test_samples]
    test_ds = RadarDataset(test_keys, [0] * len(test_keys), test_cache, augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )

    fold_models = []
    for fold in range(CFG["n_folds"]):
        m = ResNet18CBAM(CFG["n_classes"], pretrained=False, dropout=0.0).to(dev)
        ckpt = torch.load(
            out / f"best_fold{fold}.pt", map_location=dev, weights_only=True
        )
        m.load_state_dict(ckpt["model_state_dict"])
        fold_models.append(m)

    # Prediction without TTA (for comparison)
    probs_no_tta = predict_ensemble_tta(fold_models, test_loader, dev, use_tta=False)
    preds_no_tta = probs_no_tta.argmax(dim=1).numpy()

    sub_path = out / "submission_no_tta.csv"
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Predicted"])
        for k, p in zip(test_keys, preds_no_tta):
            w.writerow([k, int(p)])
    log.info("Submission (no TTA) → %s", sub_path)

    # Prediction with TTA
    log.info("Running 6-way TTA × 5 folds = 30 forward passes per sample …")
    probs_tta = predict_ensemble_tta(fold_models, test_loader, dev, use_tta=True)
    preds_tta = probs_tta.argmax(dim=1).numpy()

    sub_path = out / "submission_tta.csv"
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Predicted"])
        for k, p in zip(test_keys, preds_tta):
            w.writerow([k, int(p)])
    log.info("Submission (TTA)    → %s", sub_path)

    changed = (preds_tta != preds_no_tta).sum()
    log.info("TTA changed %d / %d predictions (%.1f%%)",
             changed, len(preds_tta), 100 * changed / len(preds_tta))
    log.info("Done.")


if __name__ == "__main__":
    main()
