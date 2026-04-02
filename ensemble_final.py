#!/usr/bin/env python3
"""
Final ensemble optimization across ALL viable models.
Evaluates geometric mean ensembles with exhaustive subset search
over the strongest models, generates top submission files.
"""
import os, csv, itertools
import torch, torch.nn.functional as F
import numpy as np
from pathlib import Path

OUT = Path("/data/slr/submissions_final2")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "v11_224":       {"path": "checkpoints_v11/eval_probs_tta.pt",      "cv": 90.46, "arch": "ConvNeXtV2", "res": 224},
    "vit_v9_224":    {"path": "checkpoints_vit_v9/eval_probs_tta.pt",   "cv": 89.50, "arch": "CAFormer",   "res": 224},
    "v18_384":       {"path": "checkpoints_v18/eval_probs_tta.pt",      "cv": 89.91, "arch": "ConvNeXtV2", "res": 384},
    "v19_288":       {"path": "checkpoints_v19/eval_probs_tta.pt",      "cv": 89.93, "arch": "ConvNeXtV2", "res": 288},
    "v20_320":       {"path": "checkpoints_v20/eval_probs_tta.pt",      "cv": 89.93, "arch": "ConvNeXtV2", "res": 320},
    "v24_384e":      {"path": "checkpoints_v24/eval_probs_tta.pt",      "cv": 88.43, "arch": "EfficientNet","res": 384},
    "v25_448":       {"path": "checkpoints_v25/eval_probs_tta.pt",      "cv": 90.21, "arch": "ConvNeXtV2", "res": 448},
    "v32_288e":      {"path": "checkpoints_v32/eval_probs_tta.pt",      "cv": 87.91, "arch": "EfficientNet","res": 288},
    "v33_448e":      {"path": "checkpoints_v33/eval_probs_tta.pt",      "cv": 88.53, "arch": "EfficientNet","res": 448},
    "v35_224conf":   {"path": "checkpoints_v35/eval_probs_tta.pt",      "cv": 90.31, "arch": "ConvNeXtV2", "res": 224},
    "v36_224mega":   {"path": "checkpoints_v36/eval_probs_tta.pt",      "cv": 90.32, "arch": "ConvNeXtV2", "res": 224},
    "v37_224mega":   {"path": "checkpoints_v37/eval_probs_tta.pt",      "cv": 89.53, "arch": "CAFormer",   "res": 224},
    "vit_v14_384":   {"path": "checkpoints_vit_v14/eval_probs_tta.pt",  "cv": 89.00, "arch": "CAFormer",   "res": 384},
    "vit_v15_288":   {"path": "checkpoints_vit_v15/eval_probs_tta.pt",  "cv": 89.09, "arch": "CAFormer",   "res": 288},
    "v14_224":       {"path": "checkpoints_v14/eval_probs_tta.pt",      "cv": 89.83, "arch": "ConvNeXtV2", "res": 224},
}

def load_probs(name, info):
    p = Path("/data/slr") / info["path"]
    if not p.exists():
        return None, None
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["keys"], d["probs"]

def geo_mean(probs_list):
    log_sum = sum(torch.log(p + 1e-8) for p in probs_list)
    return torch.exp(log_sum / len(probs_list))

def write_sub(keys, preds, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "Pred"])
        for k, p in zip(keys, preds):
            w.writerow([k.split("_", 1)[1], int(p)])

print("Loading all model probabilities...")
loaded = {}
ref_keys = None
for name, info in MODELS.items():
    keys, probs = load_probs(name, info)
    if probs is None:
        print(f"  SKIP {name} (not found)")
        continue
    if ref_keys is None:
        ref_keys = keys
    loaded[name] = probs
    print(f"  {name}: loaded ({probs.shape}), CV={info['cv']:.2f}%")

print(f"\nLoaded {len(loaded)} models")

print("\n" + "="*80)
print("PAIRWISE DISAGREEMENT ANALYSIS")
print("="*80)
names = sorted(loaded.keys())
preds = {n: loaded[n].argmax(1) for n in names}
n_samples = len(ref_keys)
for i, n1 in enumerate(names):
    for n2 in names[i+1:]:
        disagree = (preds[n1] != preds[n2]).float().mean().item()
        print(f"  {n1:20s} vs {n2:20s}: {100*disagree:.1f}% disagreement")

print("\n" + "="*80)
print("EXHAUSTIVE SUBSET SEARCH (geometric mean)")
print("="*80)

core = ["v11_224", "vit_v9_224", "v25_448", "v24_384e"]
core = [c for c in core if c in loaded]

optional = [n for n in loaded if n not in core]
print(f"Core models: {core}")
print(f"Optional models ({len(optional)}): {optional}")

results = []

for r in range(0, len(optional) + 1):
    for combo in itertools.combinations(optional, r):
        subset = list(core) + list(combo)
        if len(subset) < 3:
            continue
        probs_list = [loaded[n] for n in subset]
        ensemble_probs = geo_mean(probs_list)
        ensemble_preds = ensemble_probs.argmax(1)

        archs = set(MODELS[n]["arch"] for n in subset if n in MODELS)
        resolutions = set(MODELS[n]["res"] for n in subset if n in MODELS)

        tag = "+".join(subset)
        results.append({
            "names": subset,
            "tag": tag,
            "n_models": len(subset),
            "n_archs": len(archs),
            "n_res": len(resolutions),
        })

prev_best_path = Path("/data/slr/submissions_final/submit2_v11_vit9_v18_v25_v24_geo.csv")
prev_best_preds = None
if prev_best_path.exists():
    import pandas as pd
    df = pd.read_csv(prev_best_path)
    prev_best_preds = torch.tensor(df["Pred"].values)

print(f"\nTotal combinations evaluated: {len(results)}")

print("\n" + "="*80)
print("TOP ENSEMBLE RECOMMENDATIONS")
print("="*80)

if prev_best_preds is not None:
    for r in results:
        subset = r["names"]
        probs_list = [loaded[n] for n in subset]
        ensemble_preds = geo_mean(probs_list).argmax(1)
        diff = (ensemble_preds != prev_best_preds).float().sum().item()
        r["diff_vs_best"] = int(diff)

    results.sort(key=lambda x: (-x["n_archs"], -x["n_res"], x.get("diff_vs_best", 9999)))
else:
    results.sort(key=lambda x: (-x["n_archs"], -x["n_res"], -x["n_models"]))

seen_sigs = set()
top_n = 0
for r in results:
    if top_n >= 30:
        break
    sig = frozenset(r["names"])
    if sig in seen_sigs:
        continue
    seen_sigs.add(sig)

    diff_str = f"diff={r['diff_vs_best']}" if "diff_vs_best" in r else ""
    print(f"  [{r['n_models']}M {r['n_archs']}A {r['n_res']}R] {diff_str:10s} {' + '.join(r['names'])}")
    top_n += 1

print("\n" + "="*80)
print("GENERATING SUBMISSION FILES")
print("="*80)

submissions = [
    ("best_core4_geo", core),
    ("best5_v11_vit9_v25_v24_v18_geo", ["v11_224", "vit_v9_224", "v25_448", "v24_384e", "v18_384"]),
    ("best5_v11_vit9_v25_v24_v37_geo", ["v11_224", "vit_v9_224", "v25_448", "v24_384e", "v37_224mega"]),
    ("best6_add_v35_geo", ["v11_224", "vit_v9_224", "v25_448", "v24_384e", "v35_224conf", "v18_384"]),
    ("best6_add_v36_geo", ["v11_224", "vit_v9_224", "v25_448", "v24_384e", "v36_224mega", "v18_384"]),
    ("best6_add_v37_geo", ["v11_224", "vit_v9_224", "v25_448", "v24_384e", "v37_224mega", "v18_384"]),
    ("best7_3arch_geo", ["v11_224", "vit_v9_224", "v25_448", "v24_384e", "v18_384", "v37_224mega", "v33_448e"]),
    ("best_diverse_geo", ["v11_224", "vit_v9_224", "v25_448", "v24_384e", "v37_224mega", "v35_224conf", "v33_448e"]),
    ("mega_all_geo", list(loaded.keys())),
    ("best5_v36_v37_geo", ["v11_224", "v36_224mega", "v37_224mega", "v25_448", "v24_384e"]),
    ("best6_v35_v37_v33_geo", ["v11_224", "vit_v9_224", "v25_448", "v24_384e", "v35_224conf", "v33_448e"]),
]

for tag, subset in submissions:
    subset = [s for s in subset if s in loaded]
    if len(subset) < 2:
        print(f"  SKIP {tag} (not enough models)")
        continue
    probs_list = [loaded[n] for n in subset]
    ensemble_probs = geo_mean(probs_list)
    ensemble_preds = ensemble_probs.argmax(1).numpy()

    path = OUT / f"{tag}.csv"
    write_sub(ref_keys, ensemble_preds, path)

    if prev_best_preds is not None:
        diff = (torch.tensor(ensemble_preds) != prev_best_preds).float().sum().item()
        print(f"  {tag}: {len(subset)} models, diff_vs_89.86%={int(diff)} → {path.name}")
    else:
        print(f"  {tag}: {len(subset)} models → {path.name}")

print("\n" + "="*80)
print("ANALYSIS: Changes relative to current best (89.86% LB)")
print("="*80)
if prev_best_preds is not None:
    for tag, subset in submissions:
        subset = [s for s in subset if s in loaded]
        if len(subset) < 2: continue
        probs_list = [loaded[n] for n in subset]
        ep = geo_mean(probs_list).argmax(1)
        agree = (ep == prev_best_preds).float().mean().item()
        changed_to_new = ((ep != prev_best_preds)).float().sum().item()
        print(f"  {tag:40s}: agree={100*agree:.2f}%, changed={int(changed_to_new)}")

print("\nDone!")
