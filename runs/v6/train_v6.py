#!/usr/bin/env python3
"""
CVPR MSLR 2026 Track 2 — Italian Sign Language Recognition (Radar)
V6: ConvNeXt-Tiny + Two-Stage Training + Stochastic Depth.

Why v5 regressed (84.74% → ~83.6%):
  - ConvNeXt-Small (49.5M) overparameterized for 14.7K samples → worse
  - SWA consistently worse than best individual checkpoint
  - Disabled early stopping → model drifts past optimum
  - max_frames=56 dilutes signal for shorter sequences

V6 fixes (back to proven v4 backbone + key new technique):
  - ConvNeXt-Tiny (27.9M) — right-sized, proven at 84.74%
  - TWO-STAGE TRAINING (biggest win for ViT: 54% → 82%)
    Stage 1: head-only warmup (backbone frozen, 15 epochs)
             → head converges first, gives meaningful gradients
    Stage 2: full fine-tuning with LLRD (140 epochs, early stopping)
             → backbone adapts with proper supervision from trained head
  - Stochastic depth (drop_path_rate=0.1) — new regularization for CNN
  - max_frames=48 (proven)
  - All v4 components: GeM, MultiSampleDropout, LLRD
  - No SWA, No EMA (proven harmful with OneCycleLR)
  - Cross-architecture ensemble with v4 + ViT checkpoints

Usage:
    conda run -n slr python /data/slr/train_v6.py
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
OUT_DIR = Path("/data/slr/checkpoints_v6")
V4_DIR = Path("/data/slr/checkpoints_v4")
VIT_V3_DIR = Path("/data/slr/checkpoints_vit_v3")
VIT_V4_DIR = Path("/data/slr/checkpoints_vit_v4")
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
    "img_size": 224,
    "channel_means": [-64.3217, -62.9119, -63.9254],
    "channel_stds": [12.5721, 11.6620, 11.7259],
    # Training
    "n_folds": 5,
    "batch_size": 48,
    "num_workers": 4,
    # Stage 1: head-only warmup (backbone frozen)
    "stage1_epochs": 15,
    "stage1_lr": 5e-4,
    # Stage 2: full fine-tuning with LLRD
    "stage2_epochs": 140,
    "stage2_lr": 2e-4,
    "weight_decay": 0.05,
    "warmup_pct": 0.06,
    "label_smoothing": 0.1,
    "early_stopping_patience": 30,
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
    "drop_path_rate": 0.1,
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
#  DATASET
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
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


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


class ConvNeXtGeM(nn.Module):
    def __init__(self, model_name, n_classes=126, pretrained=True,
                 drop_path_rate=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        n_feat = self.backbone.num_features
        self.gem = GeM(p=CFG["gem_p_init"])
        self.head = MultiSampleDropout(
            n_feat, n_classes,
            n_samples=CFG["ms_dropout_samples"],
            drop_rate=CFG["drop_rate"],
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        n = sum(p.numel() for p in self.backbone.parameters())
        log.info("Backbone frozen (%d params fixed)", n)

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(True)
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info("Backbone unfrozen (%d params trainable)", n)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        pooled = self.gem(feats).flatten(1)
        return self.head(pooled)


# ===========================================================================
#  LLRD — stage-wise for ConvNeXt
# ===========================================================================


def get_param_groups(model, base_lr, stage_decay, wd):
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


def train_epoch(model, loader, optim, sched, crit, dev, use_mix=True):
    model.train()
    tot_loss, correct, n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)

        if use_mix:
            mx, ya, yb, lam = mix_or_cut(
                x, y, CFG["mixup_alpha"], CFG["cutmix_alpha"], CFG["cutmix_prob"]
            )
            logits = model(mx)
            loss = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)
        else:
            logits = model(x)
            loss = crit(logits, y)
            ya, yb, lam = y, y, 1.0

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
#  TTA
# ===========================================================================


def _tta_transforms(x: torch.Tensor) -> list[torch.Tensor]:
    views = [x]
    views.append(x.flip(dims=[2]))
    views.append(x.roll(-8, dims=2))
    views.append(x.roll(8, dims=2))
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


def _write_submission(keys, preds, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Predicted"])
        for k, p in zip(keys, preds):
            w.writerow([k, int(p)])


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
            CFG["model_name"], CFG["n_classes"], CFG["pretrained"],
            drop_path_rate=CFG["drop_path_rate"],
        ).to(dev)

        n_params = sum(p.numel() for p in model.parameters())
        log.info("Model: %s  |  %.2fM params  |  drop_path=%.1f",
                 CFG["model_name"], n_params / 1e6, CFG["drop_path_rate"])

        crit = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
        best_acc = 0.0
        ckpt_path = out / f"best_fold{fold}.pt"

        # ==== STAGE 1: Head-only warmup (backbone frozen) ====================
        log.info("── Stage 1: head-only warmup (%d epochs) ──",
                 CFG["stage1_epochs"])
        model.freeze_backbone()

        head_params = [p for p in model.parameters() if p.requires_grad]
        n_head = sum(p.numel() for p in head_params)
        log.info("Head params: %d (%.2fM)", n_head, n_head / 1e6)

        opt1 = AdamW(head_params, lr=CFG["stage1_lr"], weight_decay=0.01)
        steps1 = len(tr_loader) * CFG["stage1_epochs"]
        sched1 = torch.optim.lr_scheduler.OneCycleLR(
            opt1, max_lr=CFG["stage1_lr"], total_steps=steps1,
            pct_start=0.15, anneal_strategy="cos",
        )

        for epoch in range(1, CFG["stage1_epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(
                model, tr_loader, opt1, sched1, crit, dev, use_mix=False
            )
            va_loss, va_acc = eval_epoch(model, va_loader, crit, dev)
            dt = time.time() - t0
            log.info(
                "  S1 Ep %2d/%d  train %.4f / %.1f%%  val %.4f / %.1f%%  "
                "lr %.2e  %4.0fs",
                epoch, CFG["stage1_epochs"], tr_loss, 100 * tr_acc,
                va_loss, 100 * va_acc,
                opt1.param_groups[0]["lr"], dt,
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
        del opt1, sched1

        # ==== STAGE 2: Full fine-tuning with LLRD ============================
        log.info("── Stage 2: full fine-tuning (%d epochs, patience %d) ──",
                 CFG["stage2_epochs"], CFG["early_stopping_patience"])
        model.unfreeze_backbone()

        param_groups = get_param_groups(
            model, CFG["stage2_lr"], CFG["llrd_stage_decay"], CFG["weight_decay"]
        )
        opt2 = AdamW(param_groups)
        steps2 = len(tr_loader) * CFG["stage2_epochs"]
        sched2 = torch.optim.lr_scheduler.OneCycleLR(
            opt2,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=steps2,
            pct_start=CFG["warmup_pct"],
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        patience = 0
        for epoch in range(1, CFG["stage2_epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(
                model, tr_loader, opt2, sched2, crit, dev, use_mix=True
            )
            va_loss, va_acc = eval_epoch(model, va_loader, crit, dev)
            dt = time.time() - t0
            lr_now = opt2.param_groups[-1]["lr"]

            log.info(
                "  S2 Ep %3d/%d  train %.4f / %.1f%%  val %.4f / %.1f%%  "
                "lr %.2e  GeM_p=%.2f  %4.0fs",
                epoch, CFG["stage2_epochs"], tr_loss, 100 * tr_acc,
                va_loss, 100 * va_acc, lr_now,
                model.gem.p.item(), dt,
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

        del model, opt2, sched2, tr_loader, va_loader
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
    log.info("Loading v6 checkpoints for ensemble + TTA …")
    test_keys = [s[1] for s in test_samples]
    test_ds = RadarDataset(
        test_keys, [0] * len(test_keys), test_cache, augment=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )

    v6_models = []
    for fold in range(CFG["n_folds"]):
        m = ConvNeXtGeM(
            CFG["model_name"], CFG["n_classes"], pretrained=False
        ).to(dev)
        ckpt = torch.load(
            out / f"best_fold{fold}.pt", map_location=dev, weights_only=True
        )
        m.load_state_dict(ckpt["model_state_dict"])
        v6_models.append(m)

    probs_no = predict_ensemble_tta(v6_models, test_loader, dev, use_tta=False)
    preds_no = probs_no.argmax(dim=1).numpy()
    _write_submission(test_keys, preds_no, out / "submission_no_tta.csv")
    log.info("Submission (no TTA) → %s", out / "submission_no_tta.csv")

    log.info("Running 10-way TTA × 5 folds …")
    probs_v6 = predict_ensemble_tta(v6_models, test_loader, dev, use_tta=True)
    preds_v6 = probs_v6.argmax(dim=1).numpy()
    _write_submission(test_keys, preds_v6, out / "submission_tta.csv")
    log.info("Submission (TTA) → %s", out / "submission_tta.csv")

    changed = (preds_v6 != preds_no).sum()
    log.info("TTA changed %d / %d (%.1f%%)", changed, len(preds_v6),
             100 * changed / len(preds_v6))
    del v6_models

    # ---- Cross-architecture ensemble with v4 CNN --------------------------
    if V4_DIR.exists() and (V4_DIR / "best_fold0.pt").exists():
        log.info("Loading v4 CNN checkpoints for v4+v6 ensemble …")
        v4_models = []
        for fold in range(CFG["n_folds"]):
            m = ConvNeXtGeM(
                CFG["model_name"], CFG["n_classes"], pretrained=False
            ).to(dev)
            ckpt = torch.load(
                V4_DIR / f"best_fold{fold}.pt", map_location=dev,
                weights_only=True,
            )
            m.load_state_dict(ckpt["model_state_dict"])
            v4_models.append(m)

        log.info("Running v4 TTA …")
        probs_v4 = predict_ensemble_tta(
            v4_models, test_loader, dev, use_tta=True
        )
        del v4_models

        probs_v4v6 = 0.5 * probs_v6 + 0.5 * probs_v4
        preds_v4v6 = probs_v4v6.argmax(dim=1).numpy()
        _write_submission(
            test_keys, preds_v4v6, out / "submission_v4v6_ensemble.csv"
        )
        log.info("Submission (v4+v6 CNN ensemble) → %s",
                 out / "submission_v4v6_ensemble.csv")
        diff = (preds_v4v6 != preds_v6).sum()
        log.info("v4+v6 ensemble differs from v6-only in %d predictions", diff)
    else:
        log.info("v4 checkpoints not found — skipping CNN ensemble")

    # ---- Cross-architecture ensemble with ViT checkpoints -----------------
    vit_dir = None
    vit_model_name = None
    if VIT_V4_DIR.exists() and (VIT_V4_DIR / "best_fold0.pt").exists():
        vit_dir = VIT_V4_DIR
        vit_model_name = "swin_small_patch4_window7_224.ms_in22k_ft_in1k"
        log.info("Found ViT v4 checkpoints for CNN+ViT ensemble")
    elif VIT_V3_DIR.exists() and (VIT_V3_DIR / "best_fold0.pt").exists():
        vit_dir = VIT_V3_DIR
        vit_model_name = "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"
        log.info("Found ViT v3 checkpoints for CNN+ViT ensemble")

    if vit_dir is not None:
        log.info("Loading ViT checkpoints from %s …", vit_dir)

        class SwinGeM(nn.Module):
            def __init__(self, model_name, n_classes=126):
                super().__init__()
                self.backbone = timm.create_model(
                    model_name, pretrained=False, num_classes=0,
                )
                self.n_feat = self.backbone.num_features
                self.gem = GeM(p=CFG["gem_p_init"])
                self.head = MultiSampleDropout(
                    self.n_feat, n_classes,
                    n_samples=CFG["ms_dropout_samples"],
                    drop_rate=CFG["drop_rate"],
                )

            def forward(self, x):
                feats = self.backbone.forward_features(x)
                feats = feats.permute(0, 3, 1, 2).contiguous()
                pooled = self.gem(feats).flatten(1)
                return self.head(pooled)

        vit_models = []
        for fold in range(CFG["n_folds"]):
            ckpt_file = vit_dir / f"best_fold{fold}.pt"
            if not ckpt_file.exists():
                log.info("  ViT fold %d checkpoint missing, skipping ViT ensemble", fold)
                vit_models = []
                break
            m = SwinGeM(vit_model_name, CFG["n_classes"]).to(dev)
            ckpt = torch.load(ckpt_file, map_location=dev, weights_only=True)
            m.load_state_dict(ckpt["model_state_dict"])
            vit_models.append(m)

        if vit_models:
            log.info("Running ViT TTA …")
            probs_vit = predict_ensemble_tta(
                vit_models, test_loader, dev, use_tta=True
            )
            del vit_models

            probs_cnn_vit = 0.5 * probs_v6 + 0.5 * probs_vit
            preds_cnn_vit = probs_cnn_vit.argmax(dim=1).numpy()
            _write_submission(
                test_keys, preds_cnn_vit,
                out / "submission_cnn_vit_ensemble.csv",
            )
            log.info("Submission (CNN v6 + ViT ensemble) → %s",
                     out / "submission_cnn_vit_ensemble.csv")
            diff = (preds_cnn_vit != preds_v6).sum()
            log.info("CNN+ViT ensemble differs from CNN-only in %d predictions",
                     diff)

            # 3-way ensemble: v4 + v6 + ViT
            if V4_DIR.exists() and (V4_DIR / "best_fold0.pt").exists():
                probs_3way = (probs_v4 + probs_v6 + probs_vit) / 3.0
                preds_3way = probs_3way.argmax(dim=1).numpy()
                _write_submission(
                    test_keys, preds_3way,
                    out / "submission_3way_ensemble.csv",
                )
                log.info(
                    "Submission (v4+v6+ViT 3-way) → %s",
                    out / "submission_3way_ensemble.csv",
                )
                diff3 = (preds_3way != preds_v6).sum()
                log.info("3-way ensemble differs from v6-only in %d predictions",
                         diff3)
    else:
        log.info("No ViT checkpoints found — skipping CNN+ViT ensemble")

    log.info("Done.")


if __name__ == "__main__":
    main()
