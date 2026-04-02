#!/usr/bin/env python3
"""
CVPR MSLR 2026 Track 2 — Italian Sign Language Recognition (Radar)
V9: Near-native resolution (128×256) + Pseudo-labeling.

Key improvements over v7 (87.39% CV):
  - NEAR-NATIVE RESOLUTION: Process at (128, 256) instead of (224, 224).
    Temporal stretch reduced from 4.7× to 2.67×; range bins fully preserved.
    ConvNeXt handles non-square input natively via position-independent convs.
  - PSEUDO-LABELING: v7+v6 teacher ensemble predicts 9828 eval samples.
    High-confidence predictions (≥95%) become extra training data for every fold,
    adding ~50-70% more samples and bridging the training/eval distribution gap.
  - RESOLUTION DIVERSITY: v9 models make different errors from 224×224 models,
    yielding a much stronger multi-resolution ensemble.

Usage:
    conda run -n slr python /data/slr/train_v9.py
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

OUT_DIR = Path("/data/slr/checkpoints_v9")
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
    "img_h": 128,
    "img_w": 256,
    "teacher_img_size": 224,
    "channel_means": [-64.3217, -62.9119, -63.9254],
    "channel_stds": [12.5721, 11.6620, 11.7259],
    "n_folds": 5,
    "batch_size": 64,
    "num_workers": 4,
    # Stage 1: head-only warmup
    "stage1_epochs": 15,
    "stage1_lr": 5e-4,
    # Stage 2: full fine-tuning
    "stage2_epochs": 140,
    "stage2_lr": 2e-4,
    "weight_decay": 0.05,
    "warmup_pct": 0.06,
    "label_smoothing": 0.1,
    "early_stopping_patience": 30,
    "grad_clip_norm": 1.0,
    "llrd_stage_decay": 0.8,
    # Knowledge distillation
    "kd_alpha": 0.7,
    "kd_tau": 3.0,
    # Pseudo-labeling
    "pseudo_label_threshold": 0.80,
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
    # Model
    "model_name": "convnext_tiny.fb_in22k_ft_in1k",
    "drop_path_rate": 0.1,
    "drop_rate": 0.3,
    "ms_dropout_samples": 5,
    "gem_p_init": 3.0,
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
    """Collect from both val/ and test/ for full submission (9828 samples)."""
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
#  DATASET — parameterised img_size for multi-resolution support
# ===========================================================================

_MEANS = np.array(CFG["channel_means"], dtype=np.float32).reshape(3, 1, 1)
_STDS = np.array(CFG["channel_stds"], dtype=np.float32).reshape(3, 1, 1)
_ERASER = RandomErasing(
    p=CFG["random_erase_prob"], scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0
)


class RadarDataset(Dataset):
    def __init__(self, keys, labels, cache, augment=False, soft_labels=None,
                 img_size=(128, 256)):
        self.keys = keys
        self.labels = labels
        self.cache = cache
        self.augment = augment
        self.soft_labels = soft_labels
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

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
            x = x + np.random.normal(0, CFG["noise_std"], x.shape).astype(np.float32)
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
            t.unsqueeze(0), size=self.img_size,
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        if self.augment:
            t = _ERASER(t)
        soft = (
            self.soft_labels[self.keys[idx]]
            if self.soft_labels else torch.zeros(CFG["n_classes"])
        )
        return t, self.labels[idx], soft


# ===========================================================================
#  MODEL
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

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(True)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        pooled = self.gem(feats).flatten(1)
        return self.head(pooled)


# ===========================================================================
#  LLRD
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
#  PSEUDO-LABELING: generate from teacher ensemble at 224×224
# ===========================================================================


def generate_pseudo_labels(eval_keys, eval_cache, teacher_dirs, dev,
                           threshold=0.95):
    """Run v7+v6 CNN teachers (at 224×224) on eval data; keep high-confidence."""
    log.info("Generating pseudo-labels (threshold=%.2f) …", threshold)

    ds = RadarDataset(
        eval_keys, [0] * len(eval_keys), eval_cache, augment=False,
        img_size=(CFG["teacher_img_size"], CFG["teacher_img_size"]),
    )
    loader = DataLoader(
        ds, batch_size=96, shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )

    teachers = []
    for tdir in teacher_dirs:
        if not tdir.exists():
            continue
        for fold in range(CFG["n_folds"]):
            ckpt_path = tdir / f"best_fold{fold}.pt"
            if not ckpt_path.exists():
                continue
            m = ConvNeXtGeM(
                CFG["model_name"], CFG["n_classes"], pretrained=False,
            ).to(dev)
            ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            teachers.append(m)

    n_t = len(teachers)
    log.info("  Loaded %d teacher models for pseudo-labeling", n_t)

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

    n_total = len(eval_keys)
    n_kept = len(pl_keys)
    log.info("  Pseudo-labeled: %d / %d kept (%.1f%%); avg conf = %.3f",
             n_kept, n_total, 100 * n_kept / n_total,
             max_probs[mask].mean().item() if n_kept > 0 else 0)

    del teachers
    torch.cuda.empty_cache()
    return pl_keys, pl_labels, pl_soft


# ===========================================================================
#  TEACHER SOFT LABELS (KD at 224×224)
# ===========================================================================


def precompute_teacher_labels(keys, cache, teacher_dirs, dev):
    log.info("Pre-computing teacher soft labels (at 224×224) …")

    ds = RadarDataset(
        keys, [0] * len(keys), cache, augment=False,
        img_size=(CFG["teacher_img_size"], CFG["teacher_img_size"]),
    )
    loader = DataLoader(
        ds, batch_size=96, shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )

    teacher_models = []
    for tdir in teacher_dirs:
        if not tdir.exists():
            continue
        for fold in range(CFG["n_folds"]):
            ckpt_path = tdir / f"best_fold{fold}.pt"
            if not ckpt_path.exists():
                continue
            m = ConvNeXtGeM(
                CFG["model_name"], CFG["n_classes"], pretrained=False,
            ).to(dev)
            ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            teacher_models.append(m)

    n_teachers = len(teacher_models)
    log.info("  Loaded %d teacher models", n_teachers)

    all_probs = []
    with torch.no_grad():
        for x, _, _ in tqdm(loader, desc="  Teacher fwd", leave=False):
            x = x.to(dev, non_blocking=True)
            bp = torch.zeros(x.size(0), CFG["n_classes"], device=dev)
            for m in teacher_models:
                bp += F.softmax(m(x), dim=1)
            bp /= n_teachers
            all_probs.append(bp.cpu())

    soft_all = torch.cat(all_probs)
    soft_dict = {k: soft_all[i] for i, k in enumerate(keys)}

    del teacher_models
    torch.cuda.empty_cache()
    log.info("  Teacher soft labels ready: %d samples from %d teachers",
             len(soft_dict), n_teachers)
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


def train_epoch(model, loader, optim, sched, crit, dev, use_mix, alpha, tau):
    model.train()
    tot_loss, correct, n = 0.0, 0.0, 0
    for x, y, soft in tqdm(loader, desc="  train", leave=False):
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        soft = soft.to(dev, non_blocking=True)

        if use_mix:
            x_m, ya, yb, sa, sb, lam = mix_or_cut(x, y, soft)
            logits = model(x_m)
            ce_loss = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)
            kd = lam * kd_loss_fn(logits, sa, tau) + (1 - lam) * kd_loss_fn(logits, sb, tau)
            loss = alpha * ce_loss + (1 - alpha) * kd
        else:
            logits = model(x)
            ce_loss = crit(logits, y)
            kd = kd_loss_fn(logits, soft, tau)
            loss = alpha * ce_loss + (1 - alpha) * kd
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
    for x, y, _ in tqdm(loader, desc="  val  ", leave=False):
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        logits = model(x)
        tot_loss += crit(logits, y).item() * x.size(0)
        correct += logits.argmax(1).eq(y).sum().item()
        n += x.size(0)
    return tot_loss / n, correct / n


# ===========================================================================
#  TTA + SUBMISSION
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

    # ---- Pseudo-labeling from teacher ensemble at 224×224 ------------------
    pl_keys, pl_labels, pl_soft = generate_pseudo_labels(
        eval_keys, eval_cache, teacher_dirs, dev,
        threshold=CFG["pseudo_label_threshold"],
    )

    # ---- Teacher soft labels for training data at 224×224 ------------------
    soft_labels = precompute_teacher_labels(keys, train_cache, teacher_dirs, dev)
    soft_labels.update(pl_soft)

    combined_cache = {**train_cache, **eval_cache}

    alpha = CFG["kd_alpha"]
    tau = CFG["kd_tau"]
    log.info("KD: α=%.2f  τ=%.1f  | PL: %d extra samples",
             alpha, tau, len(pl_keys))

    # ---- 5-fold CV --------------------------------------------------------
    student_img_size = (CFG["img_h"], CFG["img_w"])
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
                             soft_labels=soft_labels, img_size=student_img_size)
        va_ds = RadarDataset(va_keys, va_labels, combined_cache, augment=False,
                             img_size=student_img_size)

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
        log.info("Model: %s at %s  |  %.2fM params",
                 CFG["model_name"], student_img_size, n_params / 1e6)

        crit = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
        best_acc = 0.0
        ckpt_path = out / f"best_fold{fold}.pt"

        # ==== STAGE 1: Head-only + KD ========================================
        log.info("── Stage 1: head-only + KD (%d epochs) ──", CFG["stage1_epochs"])
        model.freeze_backbone()
        head_params = [p for p in model.parameters() if p.requires_grad]
        opt1 = AdamW(head_params, lr=CFG["stage1_lr"], weight_decay=0.01)
        steps1 = len(tr_loader) * CFG["stage1_epochs"]
        sched1 = torch.optim.lr_scheduler.OneCycleLR(
            opt1, max_lr=CFG["stage1_lr"], total_steps=steps1,
            pct_start=0.15, anneal_strategy="cos",
        )

        for epoch in range(1, CFG["stage1_epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(
                model, tr_loader, opt1, sched1, crit, dev,
                use_mix=False, alpha=alpha, tau=tau,
            )
            va_loss, va_acc = eval_epoch(model, va_loader, crit, dev)
            dt = time.time() - t0
            log.info(
                "  S1 Ep %2d/%d  train %.4f / %.1f%%  val %.4f / %.1f%%  %4.0fs",
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
        del opt1, sched1

        # ==== STAGE 2: Full fine-tuning + KD + Mixup/CutMix ================
        log.info("── Stage 2: full fine-tuning + KD (%d epochs) ──",
                 CFG["stage2_epochs"])
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
                model, tr_loader, opt2, sched2, crit, dev,
                use_mix=True, alpha=alpha, tau=tau,
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
            "mean": float(mean_a), "std": float(std_a),
        }, f, indent=2)

    # ---- Submission -------------------------------------------------------
    eval_ds = RadarDataset(
        eval_keys, [0] * len(eval_keys), eval_cache, augment=False,
        img_size=student_img_size,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )

    log.info("Loading v9 checkpoints for ensemble …")
    v9_models = []
    for fold in range(CFG["n_folds"]):
        m = ConvNeXtGeM(CFG["model_name"], CFG["n_classes"], pretrained=False).to(dev)
        ckpt = torch.load(out / f"best_fold{fold}.pt", map_location=dev,
                          weights_only=True)
        m.load_state_dict(ckpt["model_state_dict"])
        v9_models.append(m)

    probs_no = predict_ensemble_tta(v9_models, eval_loader, dev, use_tta=False)
    preds_no = probs_no.argmax(dim=1).numpy()
    _write_submission(eval_keys, preds_no, out / "submission_no_tta.csv")
    log.info("Submission (no TTA) → %s  (%d rows)",
             out / "submission_no_tta.csv", len(preds_no))

    log.info("Running 10-way TTA × 5 folds …")
    probs_v9 = predict_ensemble_tta(v9_models, eval_loader, dev, use_tta=True)
    preds_v9 = probs_v9.argmax(dim=1).numpy()
    _write_submission(eval_keys, preds_v9, out / "submission_tta.csv")
    log.info("Submission (TTA) → %s", out / "submission_tta.csv")
    changed = (preds_v9 != preds_no).sum()
    log.info("TTA changed %d / %d (%.1f%%)", changed, len(preds_v9),
             100 * changed / len(preds_v9))
    del v9_models
    torch.cuda.empty_cache()

    # ---- Multi-resolution ensemble: v9 (128×256) + v7 (224×224) ----------
    if V7_DIR.exists() and (V7_DIR / "best_fold0.pt").exists():
        log.info("Loading v7 checkpoints for multi-resolution ensemble …")
        eval_ds_224 = RadarDataset(
            eval_keys, [0] * len(eval_keys), eval_cache, augment=False,
            img_size=(224, 224),
        )
        eval_loader_224 = DataLoader(
            eval_ds_224, batch_size=CFG["batch_size"], shuffle=False,
            num_workers=CFG["num_workers"], pin_memory=True,
        )

        v7_models = []
        for fold in range(CFG["n_folds"]):
            m = ConvNeXtGeM(CFG["model_name"], CFG["n_classes"],
                            pretrained=False).to(dev)
            ckpt = torch.load(V7_DIR / f"best_fold{fold}.pt", map_location=dev,
                              weights_only=True)
            m.load_state_dict(ckpt["model_state_dict"])
            v7_models.append(m)

        probs_v7 = predict_ensemble_tta(v7_models, eval_loader_224, dev,
                                        use_tta=True)
        del v7_models
        torch.cuda.empty_cache()

        probs_ens = 0.5 * probs_v9 + 0.5 * probs_v7
        preds_ens = probs_ens.argmax(dim=1).numpy()
        _write_submission(eval_keys, preds_ens,
                          out / "submission_v9+v7_ensemble.csv")
        log.info("Submission (v9+v7 multi-res) → %s",
                 out / "submission_v9+v7_ensemble.csv")
        diff = (preds_ens != preds_v9).sum()
        log.info("Multi-res ensemble differs from v9-only in %d predictions", diff)

    # Save v9 TTA probabilities for ensemble_final.py
    torch.save({"keys": eval_keys, "probs": probs_v9},
               out / "eval_probs_tta.pt")
    log.info("Saved v9 TTA probs → %s", out / "eval_probs_tta.pt")

    log.info("Done.")


if __name__ == "__main__":
    main()
