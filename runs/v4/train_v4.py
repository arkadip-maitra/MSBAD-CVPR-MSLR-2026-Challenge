#!/usr/bin/env python3
"""
CVPR MSLR 2026 Track 2 — Italian Sign Language Recognition (Radar)
V4: ConvNeXt-Tiny (ImageNet-22k pretrained) + GeM pooling + Multi-Sample Dropout.

Key changes vs v2 (82.4%) and v3 (82.1%):
  - ConvNeXt-Tiny backbone (28M, ImageNet-22k) replaces ResNet-18+CBAM (11M)
    → much stronger features: 7×7 kernels, LayerNorm, GELU, channel attention
  - Resize input to 224×224 (pretrained model's native resolution)
  - Repeat padding instead of zero padding (short sequences keep real signal)
  - GeM pooling (learnable generalized mean) instead of GAP
  - Multi-sample dropout (5 masks averaged during training → regularization)
  - NO EMA (v3 showed EMA+OneCycleLR is a bad combo: shadow stuck at 0.8%
    for 35 epochs while training acc was already 38%)
  - Stage-wise LLRD (per ConvNeXt stage, not per ResNet block)
  - Augmentation at v2 level (v3 proved more augmentation hurts)

Usage:
    conda run -n slr python /data/slr/train_v4.py
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

# ---------------------------------------------------------------------------
OUT_DIR = Path("/data/slr/checkpoints_v4")
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
    # Native data shape (before resize)
    "max_frames": 48,
    "n_range_bins": 256,
    "n_classes": 126,
    # Resize target
    "img_size": 224,
    # Global z-score stats
    "channel_means": [-64.3217, -62.9119, -63.9254],
    "channel_stds": [12.5721, 11.6620, 11.7259],
    # Training
    "n_folds": 5,
    "batch_size": 48,
    "num_workers": 4,
    "epochs": 120,
    "lr": 2e-4,
    "weight_decay": 0.05,
    "warmup_pct": 0.08,
    "label_smoothing": 0.1,
    "early_stopping_patience": 25,
    "grad_clip_norm": 1.0,
    # LLRD — per ConvNeXt stage
    "llrd_stage_decay": 0.8,
    # Augmentation (v2 level — proven sweet spot)
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
    # Model
    "model_name": "convnext_tiny.fb_in22k_ft_in1k",
    "drop_rate": 0.3,
    "ms_dropout_samples": 5,
    "gem_p_init": 3.0,
    "pretrained": True,
    # Misc
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
#  DATASET — augment at native res, then resize to 224×224
# ===========================================================================

_MEANS = np.array(CFG["channel_means"], dtype=np.float32).reshape(3, 1, 1)
_STDS = np.array(CFG["channel_stds"], dtype=np.float32).reshape(3, 1, 1)
_ERASER = RandomErasing(
    p=CFG["random_erase_prob"], scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0
)
_IMG_SIZE = CFG["img_size"]


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
        """Repeat-pad short sequences (keeps real signal); crop long ones."""
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
        t = F.interpolate(
            t.unsqueeze(0), size=(_IMG_SIZE, _IMG_SIZE),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        if self.augment:
            t = _ERASER(t)
        return t, self.labels[idx]


# ===========================================================================
#  MODEL COMPONENTS
# ===========================================================================


class GeM(nn.Module):
    """Generalized Mean Pooling: learnable p between avg-pool (p=1) and max-pool (p→∞)."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


class MultiSampleDropout(nn.Module):
    """Average logits from N dropout masks during training → implicit ensemble."""

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


class ConvNeXtGeM(nn.Module):
    """ConvNeXt backbone + GeM pooling + Multi-Sample Dropout head."""

    def __init__(self, n_classes=126, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            CFG["model_name"], pretrained=pretrained, num_classes=0,
        )
        n_feat = self.backbone.num_features
        self.gem = GeM(p=CFG["gem_p_init"])
        self.head = MultiSampleDropout(
            n_feat, n_classes,
            n_samples=CFG["ms_dropout_samples"],
            drop_rate=CFG["drop_rate"],
        )

    def forward(self, x):
        feats = self.backbone.forward_features(x)  # (B, 768, 7, 7) for 224×224
        pooled = self.gem(feats).flatten(1)          # (B, 768)
        return self.head(pooled)


# ===========================================================================
#  LLRD — stage-wise for ConvNeXt
# ===========================================================================


def get_param_groups(model, base_lr, stage_decay, wd):
    """Assign per-stage LR: stem < stage0 < stage1 < stage2 < stage3 < head."""
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        this_wd = wd
        if param.ndim <= 1 or "bias" in name:
            this_wd = 0.0

        if "backbone.stem" in name:
            layer_id = 0
        elif "backbone.stages.0" in name:
            layer_id = 1
        elif "backbone.stages.1" in name:
            layer_id = 2
        elif "backbone.stages.2" in name:
            layer_id = 3
        elif "backbone.stages.3" in name:
            layer_id = 4
        else:
            layer_id = 5

        scale = stage_decay ** (5 - layer_id)
        lr = base_lr * scale

        key = (lr, this_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr, "weight_decay": this_wd}
        groups[key]["params"].append(param)

    param_groups = list(groups.values())
    log.info(
        "LLRD: %d groups, LR [%.2e … %.2e]",
        len(param_groups),
        min(g["lr"] for g in param_groups),
        max(g["lr"] for g in param_groups),
    )
    return param_groups


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
#  10-WAY TTA (on 224×224 inputs)
# ===========================================================================


def _tta_transforms(x: torch.Tensor) -> list[torch.Tensor]:
    views = [x]
    views.append(x.flip(dims=[2]))                     # time-reversed
    views.append(x.roll(-8, dims=2))                   # shift up
    views.append(x.roll(8, dims=2))                    # shift down
    views.append(x[:, [1, 2, 0], :, :])                # channel perm A
    views.append(x[:, [2, 0, 1], :, :])                # channel perm B
    views.append(x + 0.05 * torch.randn_like(x))       # noise A
    views.append(x + 0.08 * torch.randn_like(x))       # noise B
    views.append(x * 0.9)                              # amplitude down
    views.append(x * 1.1)                              # amplitude up
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

        model = ConvNeXtGeM(
            n_classes=CFG["n_classes"], pretrained=CFG["pretrained"]
        ).to(dev)

        n_params = sum(p.numel() for p in model.parameters())
        log.info("Model: %s  |  %.2fM params", CFG["model_name"], n_params / 1e6)

        param_groups = get_param_groups(
            model, CFG["lr"], CFG["llrd_stage_decay"], CFG["weight_decay"]
        )
        optimizer = AdamW(param_groups)

        total_steps = len(tr_loader) * CFG["epochs"]
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=total_steps,
            pct_start=CFG["warmup_pct"],
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
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
            lr_now = optimizer.param_groups[-1]["lr"]

            log.info(
                "  Ep %3d/%d  train %.4f / %.1f%%  val %.4f / %.1f%%  "
                "lr %.2e  GeM_p=%.2f  %4.0fs",
                epoch, CFG["epochs"], tr_loss, 100 * tr_acc,
                va_loss, 100 * va_acc, lr_now,
                model.gem.p.item(), dt,
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

    # ---- Ensemble predictions with TTA ------------------------------------
    log.info("Loading best checkpoints for ensemble + TTA …")
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
        m = ConvNeXtGeM(n_classes=CFG["n_classes"], pretrained=False).to(dev)
        ckpt = torch.load(
            out / f"best_fold{fold}.pt", map_location=dev, weights_only=True
        )
        m.load_state_dict(ckpt["model_state_dict"])
        fold_models.append(m)

    probs_no = predict_ensemble_tta(fold_models, test_loader, dev, use_tta=False)
    preds_no = probs_no.argmax(dim=1).numpy()
    sub_path = out / "submission_no_tta.csv"
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Predicted"])
        for k, p in zip(test_keys, preds_no):
            w.writerow([k, int(p)])
    log.info("Submission (no TTA) → %s", sub_path)

    log.info("Running 10-way TTA × 5 folds = 50 forward passes per sample …")
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
