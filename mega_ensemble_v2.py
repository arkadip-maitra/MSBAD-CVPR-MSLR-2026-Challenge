#!/usr/bin/env python3
"""
Mega-ensemble: combine ALL viable models across architectures, resolutions, and seeds.
Generates submissions using arithmetic, geometric, and optimized weighting.
"""
import csv, torch, numpy as np
from pathlib import Path

OUT = Path("/data/slr/submissions_mega_v2")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "v11_224":    ("checkpoints_v11",    90.35),
    "vit_v9_224": ("checkpoints_vit_v9", 89.85),
    "v14_224":    ("checkpoints_v14",    89.95),
    "v18_384":    ("checkpoints_v18",    89.91),
    "v19_288":    ("checkpoints_v19",    89.93),
    "v20_320":    ("checkpoints_v20",    89.93),
    "v24_384e":   ("checkpoints_v24",    88.43),
    "v25_448":    ("checkpoints_v25",    90.21),
    "vit_v14_384":("checkpoints_vit_v14",89.00),
    "vit_v15_288":("checkpoints_vit_v15",89.09),
    "v30_224s2":  ("checkpoints_v30",    None),
    "v31_224s2":  ("checkpoints_v31",    None),
    "v26_256sw":  ("checkpoints_v26",    72.75),
}

data = {}
ref_keys = None
for name, (ckdir, cv) in MODELS.items():
    p = Path(f"/data/slr/{ckdir}/eval_probs_tta.pt")
    if not p.exists():
        print(f"  SKIP {name}: no eval_probs_tta.pt")
        continue
    d = torch.load(p, weights_only=True)
    keys, probs = d["keys"], d["probs"]
    if ref_keys is None:
        ref_keys = keys
    else:
        if keys != ref_keys:
            print(f"  WARNING: {name} has different keys, reordering...")
            key_map = {k: i for i, k in enumerate(keys)}
            idx = [key_map[k] for k in ref_keys]
            probs = probs[idx]
    data[name] = {"probs": probs, "cv": cv}
    pred = probs.argmax(1)
    print(f"  {name}: probs {tuple(probs.shape)}, cv={cv}")

print(f"\nLoaded {len(data)} models, {len(ref_keys)} eval samples")

def write_sub(keys, preds, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "Pred"])
        for k, p in zip(keys, preds):
            w.writerow([k.split("_", 1)[1], int(p)])
    print(f"  -> {path.name}")

def ensemble(names, weights=None, method="arithmetic"):
    ps = [data[n]["probs"] for n in names]
    if weights is None:
        weights = [1.0] * len(ps)
    total = sum(weights)
    weights = [w / total for w in weights]
    if method == "arithmetic":
        combined = sum(w * p for w, p in zip(weights, ps))
    elif method == "geometric":
        log_probs = sum(w * (p + 1e-8).log() for w, p in zip(weights, ps))
        combined = log_probs.exp()
        combined = combined / combined.sum(dim=1, keepdim=True)
    preds = combined.argmax(dim=1).numpy()
    return combined, preds

# Load baseline for comparison
baseline_path = Path("/data/slr/submissions_multiresolution/ensemble_v11_vitv9_v18_geometric.csv")
baseline_preds = None
if baseline_path.exists():
    with open(baseline_path) as f:
        reader = csv.reader(f)
        next(reader)
        baseline_preds = {}
        for row in reader:
            baseline_preds[row[0]] = int(row[1])

def compare_baseline(preds, name):
    if baseline_preds is None: return
    changes = sum(1 for k, p in zip(ref_keys, preds) if baseline_preds.get(k.split("_",1)[1], -1) != int(p))
    print(f"    vs baseline: {changes}/{len(ref_keys)} changed ({100*changes/len(ref_keys):.1f}%)")

# ============ ENSEMBLES ============
print("\n" + "="*60)
print("GENERATING ENSEMBLES")
print("="*60)

# 1. Core trio (our best so far)
core3 = ["v11_224", "vit_v9_224", "v18_384"]
_, p = ensemble(core3, method="geometric")
write_sub(ref_keys, p, OUT/"core3_geometric.csv")
compare_baseline(p, "core3_geo")

# 2. Core trio + v25 (448)
core4 = ["v11_224", "vit_v9_224", "v18_384", "v25_448"]
_, p = ensemble(core4, method="geometric")
write_sub(ref_keys, p, OUT/"core4_with_v25_geometric.csv")
compare_baseline(p, "core4_v25")

# 3. Full CNN ladder: v11@224 + v19@288 + v20@320 + v18@384 + v25@448
cnn_ladder = ["v11_224", "v19_288", "v20_320", "v18_384", "v25_448"]
_, p = ensemble(cnn_ladder, method="geometric")
write_sub(ref_keys, p, OUT/"cnn_ladder_5_geometric.csv")
compare_baseline(p, "cnn_ladder_5")

# 4. Full CNN ladder + CAFormer
cnn_plus_vit = cnn_ladder + ["vit_v9_224"]
_, p = ensemble(cnn_plus_vit, method="geometric")
write_sub(ref_keys, p, OUT/"cnn_ladder_plus_vit_geometric.csv")
compare_baseline(p, "cnn_plus_vit")

# 5. All high-quality models (CV > 89%)
high_q = [n for n, d in data.items() if d["cv"] is not None and d["cv"] >= 89.0]
print(f"\nHigh quality models (CV >= 89%): {high_q}")
_, p = ensemble(high_q, method="geometric")
write_sub(ref_keys, p, OUT/"all_high_quality_geometric.csv")
compare_baseline(p, "all_hq")

# 6. CV-weighted high quality
cvw = [data[n]["cv"] for n in high_q]
_, p = ensemble(high_q, weights=cvw, method="geometric")
write_sub(ref_keys, p, OUT/"all_high_quality_cvweighted_geometric.csv")
compare_baseline(p, "all_hq_cvw")

# 7. Add seed=2024 models for diversity
if "v30_224s2" in data and "v31_224s2" in data:
    diverse_all = high_q + ["v30_224s2", "v31_224s2"]
    print(f"\nDiverse all (with seed models): {diverse_all}")
    _, p = ensemble(diverse_all, method="geometric")
    write_sub(ref_keys, p, OUT/"diverse_all_geometric.csv")
    compare_baseline(p, "diverse_all")
    
    _, p = ensemble(diverse_all, method="arithmetic")
    write_sub(ref_keys, p, OUT/"diverse_all_arithmetic.csv")
    compare_baseline(p, "diverse_all_arith")

# 8. Top performers only
top = ["v11_224", "v25_448", "vit_v9_224", "v14_224"]
_, p = ensemble(top, method="geometric")
write_sub(ref_keys, p, OUT/"top4_geometric.csv")
compare_baseline(p, "top4")

# 9. Every combination of 3-5 models from high_q to find optimal subset
from itertools import combinations
print("\n--- Exhaustive search for best 3-5 model combo ---")
best_combo_names = {}
for k in range(3, min(6, len(high_q)+1)):
    best_agree = -1
    best_names = None
    for combo in combinations(high_q, k):
        _, preds = ensemble(list(combo), method="geometric")
        # Score: how many predictions agree with majority of all high_q models
        all_preds = [data[n]["probs"].argmax(1).numpy() for n in high_q]
        from collections import Counter
        agree = 0
        for i in range(len(ref_keys)):
            votes = Counter(ap[i] for ap in all_preds)
            majority = votes.most_common(1)[0][0]
            if preds[i] == majority:
                agree += 1
        if agree > best_agree:
            best_agree = agree
            best_names = combo
    if best_names:
        print(f"  Best {k}-model: {best_names} (agree={best_agree}/{len(ref_keys)})")
        _, p = ensemble(list(best_names), method="geometric")
        write_sub(ref_keys, p, OUT/f"best_{k}model_geometric.csv")
        compare_baseline(p, f"best_{k}")
        best_combo_names[k] = best_names

# 10. Majority voting ensemble
print("\n--- Majority voting ---")
all_model_names = [n for n in data.keys() if data[n]["cv"] is None or data[n]["cv"] >= 85.0]
print(f"Models for voting: {all_model_names}")
all_preds_arr = np.array([data[n]["probs"].argmax(1).numpy() for n in all_model_names])
from collections import Counter
majority_preds = []
for i in range(len(ref_keys)):
    votes = Counter(all_preds_arr[:, i])
    majority_preds.append(votes.most_common(1)[0][0])
majority_preds = np.array(majority_preds)
write_sub(ref_keys, majority_preds, OUT/"majority_vote.csv")
compare_baseline(majority_preds, "majority")

# 11. Rank-based fusion
print("\n--- Rank-based fusion ---")
rank_models = [n for n in data.keys() if data[n]["cv"] is None or data[n]["cv"] >= 89.0]
rank_sum = torch.zeros_like(data[rank_models[0]]["probs"])
for n in rank_models:
    ranks = data[n]["probs"].argsort(dim=1, descending=True).argsort(dim=1).float()
    rank_sum += ranks
rank_preds = rank_sum.argmin(dim=1).numpy()
write_sub(ref_keys, rank_preds, OUT/"rank_fusion.csv")
compare_baseline(rank_preds, "rank")

# Pairwise disagreement analysis
print("\n--- Pairwise disagreement ---")
important = ["v11_224", "vit_v9_224", "v18_384", "v25_448", "v24_384e", "v30_224s2", "v31_224s2"]
important = [n for n in important if n in data]
for i, a in enumerate(important):
    for b in important[i+1:]:
        pa = data[a]["probs"].argmax(1)
        pb = data[b]["probs"].argmax(1)
        disagree = (pa != pb).sum().item()
        print(f"  {a} vs {b}: {disagree} ({100*disagree/len(ref_keys):.1f}%)")

print(f"\nAll submissions saved to {OUT}/")
