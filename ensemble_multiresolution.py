#!/usr/bin/env python3
"""
Multi-Resolution Ensemble: combines models trained at 224×224 and 384×384.
The key insight: models at different resolutions see different aspects of the data,
so their errors are less correlated → stronger ensemble than same-resolution combos.
"""
import csv, itertools, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

OUT_DIR = Path("/data/slr/submissions_multiresolution")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "v11_224": {"path": "checkpoints_v11/eval_probs_tta.pt", "cv": 90.35, "res": 224, "arch": "convnextv2"},
    "vit_v9_224": {"path": "checkpoints_vit_v9/eval_probs_tta.pt", "cv": 89.53, "res": 224, "arch": "caformer"},
    "v14_224": {"path": "checkpoints_v14/eval_probs_tta.pt", "cv": 90.23, "res": 224, "arch": "convnextv2_sam"},
    "vit_v11_224": {"path": "checkpoints_vit_v11/eval_probs_tta.pt", "cv": 89.50, "res": 224, "arch": "caformer_sam"},
    "v18_384": {"path": "checkpoints_v18/eval_probs_tta.pt", "cv": 89.91, "res": 384, "arch": "convnextv2"},
}

base = Path("/data/slr")

print("Loading model probabilities...")
data = {}
ref_keys = None
for name, info in MODELS.items():
    p = base / info["path"]
    if not p.exists():
        print(f"  SKIP {name}: {p} not found")
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    keys = d["keys"]
    probs = d["probs"]
    if ref_keys is None:
        ref_keys = keys
    else:
        assert keys == ref_keys, f"Key mismatch for {name}"
    data[name] = {"probs": probs, **info}
    print(f"  Loaded {name}: {probs.shape} (CV={info['cv']:.2f}%, res={info['res']})")

def write_sub(keys, preds, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "Pred"])
        for k, p in zip(keys, preds):
            sid = k.split("_", 1)[1]
            w.writerow([sid, int(p)])
    print(f"  → {path} ({len(preds)} rows)")

def ensemble_models(model_names, weights=None, name_suffix=""):
    probs_list = [data[m]["probs"] for m in model_names]
    if weights is None:
        weights = [1.0] * len(probs_list)
    total = sum(weights)
    weights = [w / total for w in weights]
    combined = sum(w * p for w, p in zip(weights, probs_list))
    preds = combined.argmax(dim=1).numpy()
    model_str = "+".join(model_names)
    weight_str = "_".join(f"{w:.2f}" for w in weights)
    fname = f"ensemble_{model_str}{name_suffix}.csv"
    if len(fname) > 200:
        fname = f"ensemble_{len(model_names)}models{name_suffix}.csv"
    write_sub(ref_keys, preds, OUT_DIR / fname)
    return combined, preds

print("\n=== Multi-Resolution Ensembles ===")

# 1. Core multi-resolution: v11@224 + v18@384 (same arch, different resolution)
print("\n1. v11@224 + v18@384 (same arch, multi-res):")
ensemble_models(["v11_224", "v18_384"])

# 2. Triple: v11@224 + vit_v9@224 + v18@384 (2 archs + 2 resolutions)
print("\n2. v11@224 + vit_v9@224 + v18@384 (the multi-res triple):")
ensemble_models(["v11_224", "vit_v9_224", "v18_384"])

# 3. Same as #2 but CV-weighted
print("\n3. Same triple, CV-weighted:")
w = [MODELS["v11_224"]["cv"], MODELS["vit_v9_224"]["cv"], MODELS["v18_384"]["cv"]]
ensemble_models(["v11_224", "vit_v9_224", "v18_384"], weights=w, name_suffix="_cvw")

# 4. v11 + v18 + v14 (all ConvNeXtV2 variants at different settings)
print("\n4. All ConvNeXtV2 variants: v11 + v18 + v14:")
ensemble_models(["v11_224", "v18_384", "v14_224"])

# 5. Full 5-model ensemble
print("\n5. Full 5-model ensemble:")
ensemble_models(["v11_224", "vit_v9_224", "v18_384", "v14_224", "vit_v11_224"])

# 6. v11 + v18 CV-weighted (emphasize higher-CV model)
print("\n6. v11@224 + v18@384, CV-weighted:")
w2 = [MODELS["v11_224"]["cv"], MODELS["v18_384"]["cv"]]
ensemble_models(["v11_224", "v18_384"], weights=w2, name_suffix="_cvw")

# 7. Triple with v14_SAM + vit_v9 + v18
print("\n7. v14_SAM + vit_v9 + v18:")
ensemble_models(["v14_224", "vit_v9_224", "v18_384"])

# 8. Quad: v11 + v14 + vit_v9 + v18
print("\n8. Quad: v11 + v14 + vit_v9 + v18:")
ensemble_models(["v11_224", "v14_224", "vit_v9_224", "v18_384"])

# 9. Confidence-gated: use max-confidence model per sample
print("\n9. Confidence-gated per sample (v11 vs v18):")
p11 = data["v11_224"]["probs"]
p18 = data["v18_384"]["probs"]
conf11 = p11.max(dim=1).values
conf18 = p18.max(dim=1).values
gated = torch.where(conf11.unsqueeze(1) >= conf18.unsqueeze(1), p11, p18)
preds_gated = gated.argmax(dim=1).numpy()
write_sub(ref_keys, preds_gated, OUT_DIR / "ensemble_v11_v18_confidence_gated.csv")

# 10. Geometric mean ensemble (sharper)
print("\n10. Geometric mean ensemble (v11 + vit_v9 + v18):")
geo = (data["v11_224"]["probs"] * data["vit_v9_224"]["probs"] * data["v18_384"]["probs"]).pow(1/3)
geo = geo / geo.sum(dim=1, keepdim=True)
preds_geo = geo.argmax(dim=1).numpy()
write_sub(ref_keys, preds_geo, OUT_DIR / "ensemble_v11_vitv9_v18_geometric.csv")

# Compare predictions across ensembles
print("\n=== Disagreement Analysis ===")
base_preds = data["v11_224"]["probs"].argmax(1).numpy()
v18_preds = data["v18_384"]["probs"].argmax(1).numpy()
vit_preds = data["vit_v9_224"]["probs"].argmax(1).numpy()
disagree_v11_v18 = (base_preds != v18_preds).sum()
disagree_v11_vit = (base_preds != vit_preds).sum()
disagree_v18_vit = (v18_preds != vit_preds).sum()
print(f"v11 vs v18 disagree: {disagree_v11_v18} / {len(base_preds)} ({100*disagree_v11_v18/len(base_preds):.1f}%)")
print(f"v11 vs vit_v9 disagree: {disagree_v11_vit} / {len(base_preds)} ({100*disagree_v11_vit/len(base_preds):.1f}%)")
print(f"v18 vs vit_v9 disagree: {disagree_v18_vit} / {len(base_preds)} ({100*disagree_v18_vit/len(base_preds):.1f}%)")

print("\nDone! Check submissions in", OUT_DIR)
