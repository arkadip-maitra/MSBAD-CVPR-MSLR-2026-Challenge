#!/usr/bin/env python3
"""
Optimized weighted ensemble for the 6-model best ensemble.
Instead of equal-weight geometric mean, find optimal per-model weights
using power-mean and grid search approaches.
"""
import torch, csv, itertools
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import os

BEST_6 = {
    "v11_224":    ("checkpoints_v11",     90.46, "convnextv2_tiny"),
    "vit_v9_224": ("checkpoints_vit_v9",  89.50, "caformer_s18"),
    "v18_384":    ("checkpoints_v18",     89.91, "convnextv2_tiny"),
    "v25_448":    ("checkpoints_v25",     90.21, "convnextv2_tiny"),
    "v24_384e":   ("checkpoints_v24",     88.43, "efficientnetv2_s"),
    "v38_512":    ("checkpoints_v38",     89.89, "convnextv2_tiny"),
}

OUT = Path("/data/slr/submissions_weighted")
OUT.mkdir(exist_ok=True)

def load_eval(name, info):
    d = torch.load(f"/data/slr/{info[0]}/eval_probs_tta.pt", map_location="cpu", weights_only=True)
    return d["keys"], d["probs"]

def load_oof_fold_probs(ckpt_dir, fold, model_cls_name):
    """Load checkpoint for a fold and return it exists."""
    cp = Path(f"/data/slr/{ckpt_dir}/best_fold{fold}.pt")
    return cp.exists()

def weighted_geo(probs_list, weights):
    """Weighted geometric mean: exp(sum(w_i * log(p_i)) / sum(w_i))"""
    w = np.array(weights)
    w = w / w.sum()
    log_sum = sum(w[i] * torch.log(probs_list[i] + 1e-8) for i in range(len(probs_list)))
    return torch.exp(log_sum)

def power_mean(probs_list, p=1.0):
    """Generalized power mean of probabilities."""
    if abs(p) < 1e-8:
        return torch.exp(sum(torch.log(pr + 1e-8) for pr in probs_list) / len(probs_list))
    stacked = torch.stack(probs_list)
    return (stacked.pow(p).mean(0)).pow(1.0 / p)

def write_sub(keys, preds, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "Pred"])
        for k, p in zip(keys, preds):
            w.writerow([k.split("_", 1)[1], int(p)])

print("Loading eval probabilities...")
eval_probs = {}
ref_keys = None
for name, info in BEST_6.items():
    keys, probs = load_eval(name, info)
    if ref_keys is None: ref_keys = keys
    eval_probs[name] = probs
    print(f"  {name}: {probs.shape}")

names = list(BEST_6.keys())
probs_list = [eval_probs[n] for n in names]

import pandas as pd
prev = pd.read_csv("/data/slr/submissions_final3/submit1_best5_plus_v38_geo.csv")
prev_preds = torch.tensor(prev["Pred"].values)

print("\n" + "="*80)
print("1. POWER MEAN SEARCH")
print("="*80)
for p in [-2, -1, -0.5, 0, 0.5, 1, 2, 3, 5]:
    pm = power_mean(probs_list, p)
    preds = pm.argmax(1)
    diff = (preds != prev_preds).float().sum().item()
    label = "geometric" if abs(p) < 0.01 else f"p={p}"
    print(f"  {label:12s}: changed={int(diff):4d} vs 89.906% LB")
    if abs(p) < 0.01:
        path = OUT / "geo_equal.csv"
    else:
        path = OUT / f"power_p{p}.csv"
    write_sub(ref_keys, preds.numpy(), path)

print("\n" + "="*80)
print("2. CV-WEIGHTED GEOMETRIC MEAN")
print("="*80)
cvs = [BEST_6[n][1] for n in names]
for temp in [0.5, 1.0, 2.0, 5.0, 10.0]:
    w = np.array(cvs)
    w = np.exp((w - w.mean()) * temp / w.std())
    wg = weighted_geo(probs_list, w)
    preds = wg.argmax(1)
    diff = (preds != prev_preds).float().sum().item()
    print(f"  temp={temp:5.1f}: weights=[{', '.join(f'{x:.2f}' for x in w/w.sum())}] changed={int(diff)}")
    write_sub(ref_keys, preds.numpy(), OUT / f"cv_weighted_t{temp}.csv")

print("\n" + "="*80)
print("3. LEAVE-ONE-OUT ANALYSIS")
print("="*80)
for i, name in enumerate(names):
    subset = [probs_list[j] for j in range(len(names)) if j != i]
    eg = torch.exp(sum(torch.log(p + 1e-8) for p in subset) / len(subset))
    preds = eg.argmax(1)
    diff = (preds != prev_preds).float().sum().item()
    print(f"  Drop {name:12s} (CV={cvs[i]:.1f}%): changed={int(diff)}")

print("\n" + "="*80)
print("4. CONFIDENCE-WEIGHTED GEOMETRIC MEAN")
print("="*80)
for threshold in [0.7, 0.8, 0.9, 0.95]:
    weighted_probs = torch.zeros_like(probs_list[0])
    total_weight = torch.zeros(probs_list[0].size(0), 1)
    for p in probs_list:
        conf = p.max(1, keepdim=True).values
        w = (conf > threshold).float() + 0.5 * (conf <= threshold).float()
        weighted_probs += w * torch.log(p + 1e-8)
        total_weight += w
    result = torch.exp(weighted_probs / total_weight)
    preds = result.argmax(1)
    diff = (preds != prev_preds).float().sum().item()
    print(f"  threshold={threshold:.2f}: changed={int(diff)}")
    write_sub(ref_keys, preds.numpy(), OUT / f"conf_weighted_{threshold}.csv")

print("\n" + "="*80)
print("5. SHARPENED GEOMETRIC MEAN (temperature scaling)")
print("="*80)
base_geo = torch.exp(sum(torch.log(p + 1e-8) for p in probs_list) / len(probs_list))
for temp in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    sharpened = F.softmax(torch.log(base_geo + 1e-8) / temp, dim=1)
    preds = sharpened.argmax(1)
    diff = (preds != prev_preds).float().sum().item()
    print(f"  temp={temp:.1f}: changed={int(diff)}")
    if temp != 1.0:
        write_sub(ref_keys, preds.numpy(), OUT / f"sharp_t{temp}.csv")

print("\n" + "="*80)
print("6. BEST SUBSET OF 5 (drop weakest contributor)")
print("="*80)
for drop_name in names:
    subset_names = [n for n in names if n != drop_name]
    subset_probs = [eval_probs[n] for n in subset_names]
    eg = torch.exp(sum(torch.log(p + 1e-8) for p in subset_probs) / len(subset_probs))
    preds = eg.argmax(1)
    diff = (preds != prev_preds).float().sum().item()
    tag = "+".join(subset_names)
    print(f"  Drop {drop_name:12s}: {len(subset_names)} models, changed={int(diff)}")
    write_sub(ref_keys, preds.numpy(), OUT / f"drop_{drop_name}.csv")

print("\nDone! All files in", OUT)
