#!/usr/bin/env python3
"""
v12c: EfficientNetV2-S + SAM (Sharpness-Aware Minimization)
Born-again from v12, NO pseudo-labels, cross-arch KD from v11 + vit_v9

Key changes vs v12b:
  1. SAM optimizer for flat minima → better generalization (closes CV-to-LB gap)
  2. ZERO pseudo-labels — removes eval-set feedback loop that caused overfitting
  3. Born-again from v12 (original, not the overfit v12b)
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

OUT_DIR = Path("/data/slr/checkpoints_v12c")
V12_DIR = Path("/data/slr/checkpoints_v12")
V11_DIR = Path("/data/slr/checkpoints_v11")
VIT_V9_DIR = Path("/data/slr/checkpoints_vit_v9")
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
    "n_classes": 126,
    "channel_means": [-64.3217, -62.9119, -63.9254],
    "channel_stds": [12.5721, 11.6620, 11.7259],
    "n_folds": 5,
    "batch_size": 48,
    "num_workers": 4,
    "img_size": 224,
    "stage1_epochs": 5,
    "stage1_lr": 2e-4,
    "stage2_epochs": 100,
    "stage2_lr": 1e-5,
    "weight_decay": 0.05,
    "warmup_pct": 0.15,
    "label_smoothing": 0.1,
    "early_stopping_patience": 40,
    "grad_clip_norm": 1.0,
    "llrd_decay": 0.85,
    "kd_alpha": 0.5,
    "kd_tau": 4.0,
    "sam_rho": 0.05,
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
    "model_name": "tf_efficientnetv2_s.in21k_ft_in1k",
    "drop_path_rate": 0.2,
    "proj_dim": 384,
    "drop_rate": 0.3,
    "ms_dropout_samples": 5,
    "seed": 42,
}


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


_MEANS = np.array(CFG["channel_means"], dtype=np.float32).reshape(3, 1, 1)
_STDS = np.array(CFG["channel_stds"], dtype=np.float32).reshape(3, 1, 1)
_ERASER = RandomErasing(
    p=CFG["random_erase_prob"], scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0
)


class RadarDataset(Dataset):
    def __init__(self, keys, labels, cache, augment=False, soft_labels=None,
                 img_size=224):
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
        if np.random.random() < CFG["circ_shift_prob"]:
            shift = np.random.randint(-CFG["circ_shift_max"],
                                       CFG["circ_shift_max"] + 1)
            x = np.roll(x, shift, axis=1)
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
        sz = self.img_size
        if isinstance(sz, int):
            sz = (sz, sz)
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


# ---------------------------------------------------------------------------
#  MODELS
# ---------------------------------------------------------------------------

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


class MultiScaleEfficientNetV2(nn.Module):
    def __init__(self, model_name, n_classes=126, pretrained=True,
                 drop_path_rate=0.2, proj_dim=384, drop_rate=0.3, ms_samples=5):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True,
            out_indices=(0, 1, 2, 3, 4), drop_path_rate=drop_path_rate,
        )
        stage_dims = self.backbone.feature_info.channels()
        self.stage_gems = nn.ModuleList([GeM(p=3.0) for _ in stage_dims])
        self.stage_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
            ) for d in stage_dims
        ])
        self.scale_attn = nn.Sequential(nn.Linear(proj_dim, 1))
        self.head = MultiSampleDropout(proj_dim, n_classes,
                                       n_samples=ms_samples, drop_rate=drop_rate)

    def forward(self, x):
        feats = self.backbone(x)
        projs = [proj(gem(f).flatten(1))
                 for gem, proj, f in zip(self.stage_gems, self.stage_projs, feats)]
        stacked = torch.stack(projs, dim=1)
        weights = F.softmax(self.scale_attn(stacked), dim=1)
        aggr = (stacked * weights).sum(dim=1)
        return self.head(aggr)


class MultiScaleConvNeXtV2(nn.Module):
    def __init__(self, model_name, n_classes=126, pretrained=True,
                 drop_path_rate=0.0, proj_dim=512, drop_rate=0.3, ms_samples=5):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True,
            out_indices=(0, 1, 2, 3), drop_path_rate=drop_path_rate,
        )
        stage_dims = self.backbone.feature_info.channels()
        self.stage_gems = nn.ModuleList([GeM(p=3.0) for _ in stage_dims])
        self.stage_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
            ) for d in stage_dims
        ])
        self.scale_attn = nn.Sequential(nn.Linear(proj_dim, 1))
        self.head = MultiSampleDropout(proj_dim, n_classes,
                                       n_samples=ms_samples, drop_rate=drop_rate)

    def forward(self, x):
        feats = self.backbone(x)
        projs = [proj(gem(f).flatten(1))
                 for gem, proj, f in zip(self.stage_gems, self.stage_projs, feats)]
        stacked = torch.stack(projs, dim=1)
        weights = F.softmax(self.scale_attn(stacked), dim=1)
        aggr = (stacked * weights).sum(dim=1)
        return self.head(aggr)


class MultiScaleCAFormer(nn.Module):
    def __init__(self, model_name, n_classes=126, pretrained=True,
                 drop_path_rate=0.0, proj_dim=384, drop_rate=0.3, ms_samples=5):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True,
            out_indices=(0, 1, 2, 3), drop_path_rate=drop_path_rate,
        )
        stage_dims = self.backbone.feature_info.channels()
        self.stage_gems = nn.ModuleList([GeM(p=3.0) for _ in stage_dims])
        self.stage_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
            ) for d in stage_dims
        ])
        self.scale_attn = nn.Sequential(nn.Linear(proj_dim, 1))
        self.head = MultiSampleDropout(proj_dim, n_classes,
                                       n_samples=ms_samples, drop_rate=drop_rate)

    def forward(self, x):
        feats = self.backbone(x)
        projs = [proj(gem(f).flatten(1))
                 for gem, proj, f in zip(self.stage_gems, self.stage_projs, feats)]
        stacked = torch.stack(projs, dim=1)
        weights = F.softmax(self.scale_attn(stacked), dim=1)
        aggr = (stacked * weights).sum(dim=1)
        return self.head(aggr)


# ---------------------------------------------------------------------------
#  SAM — Sharpness-Aware Minimization
# ---------------------------------------------------------------------------

class SAM:
    """Wraps an existing optimizer to add SAM perturbation step."""
    def __init__(self, base_optimizer, rho=0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho
        self._eps = {}

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self._eps[p] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p in self._eps:
                    p.sub_(self._eps[p])
        self.base_optimizer.step()
        self._eps.clear()

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        norms = []
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.detach().norm(p=2))
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), p=2)


# ---------------------------------------------------------------------------
#  TEACHER CONFIG (KD on training data only — no pseudo-labels)
# ---------------------------------------------------------------------------

TEACHER_CONFIGS = [
    {"dir": V11_DIR, "cls": MultiScaleConvNeXtV2,
     "model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
     "kwargs": {"drop_path_rate": 0.0, "proj_dim": 512, "drop_rate": 0.3,
                "ms_samples": 5}},
    {"dir": VIT_V9_DIR, "cls": MultiScaleCAFormer,
     "model_name": "caformer_s18.sail_in22k_ft_in1k",
     "kwargs": {"drop_path_rate": 0.0, "proj_dim": 384, "drop_rate": 0.3,
                "ms_samples": 5}},
]


def load_born_again(model, ckpt_path, dev):
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    val_acc = ckpt.get("val_acc", 0.0)
    log.info("  Born-again ← %s (val %.2f%%)", ckpt_path.name, 100 * val_acc)
    return val_acc


def get_llrd_params(model, base_lr, decay=0.85, wd=0.05):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone.conv_stem" in name or "backbone.bn1" in name:
            lr = base_lr * decay ** 6
        elif "backbone.blocks.0." in name:
            lr = base_lr * decay ** 5
        elif "backbone.blocks.1." in name:
            lr = base_lr * decay ** 4
        elif "backbone.blocks.2." in name:
            lr = base_lr * decay ** 3
        elif "backbone.blocks.3." in name:
            lr = base_lr * decay ** 2
        elif "backbone.blocks.4." in name:
            lr = base_lr * decay ** 1
        elif "backbone.blocks.5." in name:
            lr = base_lr
        else:
            lr = base_lr
        w = 0.0 if p.ndim < 2 else wd
        key = (lr, w)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr, "weight_decay": w}
        groups[key]["params"].append(p)
    result = list(groups.values())
    lrs = sorted(set(g["lr"] for g in result))
    log.info("LLRD: %d groups, LR [%.2e … %.2e]", len(result), min(lrs), max(lrs))
    return result


def _load_teachers_from_config(configs, dev):
    teachers = []
    for cfg in configs:
        tdir = cfg["dir"]
        if not tdir.exists():
            continue
        cls = cfg["cls"]
        kwargs = {"n_classes": CFG["n_classes"], "pretrained": False,
                  **cfg["kwargs"]}
        for fold in range(CFG["n_folds"]):
            ckpt_path = tdir / f"best_fold{fold}.pt"
            if not ckpt_path.exists():
                continue
            m = cls(cfg["model_name"], **kwargs).to(dev)
            ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            teachers.append(m)
    return teachers


def _teacher_infer(teachers, loader, dev):
    all_probs = []
    n_t = len(teachers)
    with torch.no_grad():
        for x, _, _ in tqdm(loader, desc="  teacher fwd", leave=False):
            x = x.to(dev, non_blocking=True)
            bp = torch.zeros(x.size(0), CFG["n_classes"], device=dev)
            for m in teachers:
                bp += F.softmax(m(x), dim=1)
            bp /= n_t
            all_probs.append(bp.cpu())
    return torch.cat(all_probs)


def precompute_teacher_labels(keys, cache, teacher_configs, dev):
    log.info("Pre-computing teacher soft labels on TRAINING data only …")
    ds = RadarDataset(keys, [0] * len(keys), cache, augment=False,
                      img_size=CFG["img_size"])
    loader = DataLoader(ds, batch_size=96, shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True)
    teachers = _load_teachers_from_config(teacher_configs, dev)
    log.info("  Loaded %d teacher models", len(teachers))
    soft_all = _teacher_infer(teachers, loader, dev)
    soft_dict = {k: soft_all[i] for i, k in enumerate(keys)}
    del teachers
    torch.cuda.empty_cache()
    log.info("  Soft labels ready: %d samples (training only, no PL)", len(soft_dict))
    return soft_dict


# ---------------------------------------------------------------------------
#  MIXUP / CUTMIX / KD
# ---------------------------------------------------------------------------

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


def kd_loss_fn(logits, soft_target, tau):
    log_pred = F.log_softmax(logits / tau, dim=1)
    soft_tau = (soft_target + 1e-8).pow(1.0 / tau)
    soft_tau = soft_tau / soft_tau.sum(dim=1, keepdim=True)
    return F.kl_div(log_pred, soft_tau, reduction="batchmean") * (tau ** 2)


def _compute_loss(model, x_m, ya, yb, sa, sb, lam, crit, alpha, tau):
    logits = model(x_m)
    ce = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)
    kd = lam * kd_loss_fn(logits, sa, tau) + (1 - lam) * kd_loss_fn(logits, sb, tau)
    loss = alpha * ce + (1 - alpha) * kd
    return logits, loss


# ---------------------------------------------------------------------------
#  TRAIN / EVAL  (with SAM support)
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, sched, crit, dev, alpha, tau,
                use_mix=True, is_sam=False):
    model.train()
    tot_loss, correct, n = 0.0, 0.0, 0
    for x, y, soft in tqdm(loader, desc="  train", leave=False):
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        soft = soft.to(dev, non_blocking=True)

        if use_mix:
            x_m, ya, yb, sa, sb, lam = mix_or_cut(x, y, soft)
        else:
            x_m, ya, yb, sa, sb, lam = x, y, y, soft, soft, 1.0

        logits, loss = _compute_loss(model, x_m, ya, yb, sa, sb, lam,
                                     crit, alpha, tau)
        loss.backward()

        if is_sam:
            optimizer.first_step()
            optimizer.zero_grad(set_to_none=True)
            _, loss2 = _compute_loss(model, x_m, ya, yb, sa, sb, lam,
                                     crit, alpha, tau)
            loss2.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip_norm"])
            optimizer.second_step()
            optimizer.zero_grad(set_to_none=True)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

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


def _tta_transforms(x):
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


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

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

    soft_labels = precompute_teacher_labels(keys, train_cache, TEACHER_CONFIGS, dev)

    alpha = CFG["kd_alpha"]
    tau = CFG["kd_tau"]
    log.info("KD: α=%.2f  τ=%.1f  | SAM ρ=%.3f  | NO pseudo-labels",
             alpha, tau, CFG["sam_rho"])

    probe = MultiScaleEfficientNetV2(
        CFG["model_name"], CFG["n_classes"], pretrained=False,
        drop_path_rate=CFG["drop_path_rate"], proj_dim=CFG["proj_dim"],
        drop_rate=CFG["drop_rate"], ms_samples=CFG["ms_dropout_samples"],
    )
    n_params = sum(p.numel() for p in probe.parameters())
    log.info("MultiScaleEfficientNetV2: %.2fM params", n_params / 1e6)
    del probe

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

        log.info("  Train: %d (labeled only, no PL)  |  Val: %d",
                 len(tr_keys), len(va_keys))

        tr_ds = RadarDataset(tr_keys, tr_labels, train_cache, augment=True,
                             soft_labels=soft_labels, img_size=CFG["img_size"])
        va_ds = RadarDataset(va_keys, va_labels, train_cache, augment=False,
                             img_size=CFG["img_size"])

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

        model = MultiScaleEfficientNetV2(
            CFG["model_name"], CFG["n_classes"], pretrained=True,
            drop_path_rate=CFG["drop_path_rate"], proj_dim=CFG["proj_dim"],
            drop_rate=CFG["drop_rate"], ms_samples=CFG["ms_dropout_samples"],
        ).to(dev)

        born_again_path = V12_DIR / f"best_fold{fold}.pt"
        if born_again_path.exists():
            load_born_again(model, born_again_path, dev)
        else:
            log.info("  No v12 checkpoint for fold %d, using ImageNet init", fold)

        crit = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
        best_acc = 0.0
        ckpt_path = out / f"best_fold{fold}.pt"

        # ── Stage 1: head-only warmup (no SAM) ──
        log.info("── Stage 1: head-only warmup (%d epochs, no mix, no SAM) ──",
                 CFG["stage1_epochs"])

        for p in model.backbone.parameters():
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
                use_mix=False, is_sam=False,
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

        # ── Stage 2: full fine-tune + SAM + LLRD ──
        log.info("── Stage 2: full fine-tune + SAM + LLRD (%d epochs, patience %d) ──",
                 CFG["stage2_epochs"], CFG["early_stopping_patience"])

        for p in model.parameters():
            p.requires_grad = True

        param_groups = get_llrd_params(
            model, CFG["stage2_lr"],
            decay=CFG["llrd_decay"], wd=CFG["weight_decay"],
        )
        base_opt = AdamW(param_groups, lr=CFG["stage2_lr"],
                         weight_decay=CFG["weight_decay"])
        sam_opt = SAM(base_opt, rho=CFG["sam_rho"])
        steps = len(tr_loader) * CFG["stage2_epochs"]
        sched = torch.optim.lr_scheduler.OneCycleLR(
            base_opt, max_lr=[g["lr"] for g in param_groups],
            total_steps=steps, pct_start=CFG["warmup_pct"],
            anneal_strategy="cos",
            div_factor=25.0, final_div_factor=1000.0,
        )

        patience = 0
        for epoch in range(1, CFG["stage2_epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(
                model, tr_loader, sam_opt, sched, crit, dev, alpha, tau,
                use_mix=True, is_sam=True,
            )
            va_loss, va_acc = eval_epoch(model, va_loader, crit, dev)
            dt = time.time() - t0
            lr_now = base_opt.param_groups[-1]["lr"]
            gem_p = model.stage_gems[-1].p.item()
            log.info(
                "  S2 Ep %3d/%d  train %.4f / %.1f%%  val %.4f / %.1f%%  "
                "lr %.2e  GeM_p=%.2f  %4.0fs",
                epoch, CFG["stage2_epochs"], tr_loss, 100 * tr_acc,
                va_loss, 100 * va_acc, lr_now, gem_p, dt,
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
        del model, base_opt, sam_opt, sched, tr_loader, va_loader
        torch.cuda.empty_cache()

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

    eval_ds = RadarDataset(eval_keys, [0] * len(eval_keys), eval_cache,
                           augment=False, img_size=CFG["img_size"])
    eval_loader = DataLoader(
        eval_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )

    log.info("Loading v12c checkpoints for ensemble …")
    fold_models = []
    for fold in range(CFG["n_folds"]):
        m = MultiScaleEfficientNetV2(
            CFG["model_name"], CFG["n_classes"], pretrained=False,
            drop_path_rate=0.0, proj_dim=CFG["proj_dim"],
            drop_rate=CFG["drop_rate"], ms_samples=CFG["ms_dropout_samples"],
        ).to(dev)
        ckpt = torch.load(out / f"best_fold{fold}.pt", map_location=dev,
                          weights_only=True)
        m.load_state_dict(ckpt["model_state_dict"])
        fold_models.append(m)

    probs_no = predict_ensemble_tta(fold_models, eval_loader, dev, use_tta=False)
    preds_no = probs_no.argmax(dim=1).numpy()
    _write_submission(eval_keys, preds_no, out / "submission_no_tta.csv")
    log.info("Submission (no TTA) → %s  (%d rows)",
             out / "submission_no_tta.csv", len(preds_no))

    log.info("Running 10-way TTA × 5 folds …")
    probs_tta = predict_ensemble_tta(fold_models, eval_loader, dev, use_tta=True)
    preds_tta = probs_tta.argmax(dim=1).numpy()
    _write_submission(eval_keys, preds_tta, out / "submission_tta.csv")
    log.info("Submission (TTA) → %s", out / "submission_tta.csv")
    changed = (preds_tta != preds_no).sum()
    log.info("TTA changed %d / %d (%.1f%%)", changed, len(preds_tta),
             100 * changed / len(preds_tta))

    torch.save({"keys": eval_keys, "probs": probs_tta},
               out / "eval_probs_tta.pt")
    log.info("Saved v12c TTA probs → %s", out / "eval_probs_tta.pt")

    log.info("Done.")


if __name__ == "__main__":
    main()
