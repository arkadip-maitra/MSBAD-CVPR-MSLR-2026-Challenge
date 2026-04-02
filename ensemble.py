#!/usr/bin/env python3
"""
Hybrid ensemble: preserve 6-model geo-mean for confident predictions,
selectively incorporate CAFormer diversity only where uncertainty is high.

Also: try confidence-gated voting, rank fusion, and calibrated ensembles.
"""
import torch, csv, numpy as np
import torch.nn.functional as F
from pathlib import Path
import pandas as pd

MODELS_6 = {
    "v11_224":    "checkpoints_v11",
    "vit_v9_224": "checkpoints_vit_v9",
    "v18_384":    "checkpoints_v18",
    "v25_448":    "checkpoints_v25",
    "v24_384e":   "checkpoints_v24",
    "v38_512":    "checkpoints_v38",
}

MODELS_CAF = {
    "v43_320":    "checkpoints_v43",
    "v45_384":    "checkpoints_v45",
}

OUT = Path("/data/slr/submissions_final5")
OUT.mkdir(exist_ok=True)

def load_eval(ckpt_dir):
    d = torch.load(f"/data/slr/{ckpt_dir}/eval_probs_tta.pt", map_location="cpu", weights_only=True)
    return d["keys"], d["probs"]

def write_sub(keys, preds, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "Pred"])
        for k, p in zip(keys, preds):
            w.writerow([k.split("_", 1)[1], int(p)])

print("Loading probabilities...")
ref_keys = None
probs_6 = {}
for name, ckpt in MODELS_6.items():
    keys, probs = load_eval(ckpt)
    if ref_keys is None: ref_keys = keys
    probs_6[name] = probs
    print(f"  {name}: {probs.shape}")

probs_caf = {}
for name, ckpt in MODELS_CAF.items():
    keys, probs = load_eval(ckpt)
    probs_caf[name] = probs
    print(f"  {name}: {probs.shape}")

base_lb = pd.read_csv("submissions_final3/submit1_best5_plus_v38_geo.csv")
base_preds = torch.tensor(base_lb["Pred"].values)

all_6 = list(probs_6.values())
all_caf = list(probs_caf.values())
all_8 = all_6 + all_caf

geo6 = torch.exp(sum(torch.log(p + 1e-8) for p in all_6) / len(all_6))
geo8 = torch.exp(sum(torch.log(p + 1e-8) for p in all_8) / len(all_8))

geo6_preds = geo6.argmax(1)
geo6_conf = geo6.max(1).values

n_samples = geo6.size(0)

def report(name, preds):
    diff = (preds != base_preds).sum().item()
    print(f"  {name:55s}: {int(diff):4d} changes vs 89.906%")
    write_sub(ref_keys, preds.numpy(), OUT / f"{name}.csv")

print("\n" + "="*80)
print("STRATEGY 1: Confidence-gated CAFormer injection")
print("Only use 8-model geo mean where 6-model confidence < threshold")
print("="*80)
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    uncertain = geo6_conf < thresh
    hybrid_preds = geo6_preds.clone()
    hybrid_preds[uncertain] = geo8.argmax(1)[uncertain]
    n_uncertain = uncertain.sum().item()
    report(f"hybrid_inject_t{thresh}_(n={n_uncertain})", hybrid_preds)

print("\n" + "="*80)
print("STRATEGY 2: Majority voting with geo-mean tiebreak")
print("Use prediction that gets most votes; tiebreak by geo-mean confidence")
print("="*80)
for model_set, name_prefix, probs_list in [
    ("6-model", "vote6", all_6),
    ("8-model", "vote8", all_8),
]:
    preds_per_model = torch.stack([p.argmax(1) for p in probs_list])  # (M, N)
    n_models = preds_per_model.size(0)
    n_classes = probs_list[0].size(1)

    vote_counts = torch.zeros(n_samples, n_classes, dtype=torch.long)
    for m in range(n_models):
        for i in range(n_samples):
            vote_counts[i, preds_per_model[m, i]] += 1

    max_votes = vote_counts.max(1).values
    vote_preds = torch.zeros(n_samples, dtype=torch.long)

    geo_ref = geo6 if model_set == "6-model" else geo8
    for i in range(n_samples):
        candidates = (vote_counts[i] == max_votes[i]).nonzero(as_tuple=True)[0]
        if len(candidates) == 1:
            vote_preds[i] = candidates[0]
        else:
            best_c = candidates[geo_ref[i, candidates].argmax()]
            vote_preds[i] = best_c

    report(f"{name_prefix}_majority_vote", vote_preds)

print("\n" + "="*80)
print("STRATEGY 3: Per-model confidence weighting (sample-level)")
print("Each model's weight = its softmax entropy (low entropy = high weight)")
print("="*80)
for model_set, name_prefix, probs_list in [
    ("6-model", "entropy6", all_6),
    ("8-model", "entropy8", all_8),
]:
    log_sum = torch.zeros_like(probs_list[0])
    total_weight = torch.zeros(n_samples, 1)

    for p in probs_list:
        entropy = -(p * torch.log(p + 1e-8)).sum(1, keepdim=True)
        max_entropy = np.log(p.size(1))
        weight = 1.0 - (entropy / max_entropy)
        weight = weight.clamp(min=0.1)
        log_sum += weight * torch.log(p + 1e-8)
        total_weight += weight

    result = torch.exp(log_sum / total_weight)
    preds = result.argmax(1)
    report(f"{name_prefix}_entropy_weighted", preds)

print("\n" + "="*80)
print("STRATEGY 4: Top-K agreement filter")
print("Only change from baseline when K strongest models (by conf) agree")
print("="*80)
for k_agree in [4, 5, 6]:
    preds_per = [p.argmax(1) for p in all_6]
    conf_per = [p.max(1).values for p in all_6]

    hybrid_preds = geo6_preds.clone()
    n_changed = 0
    for i in range(n_samples):
        confs = torch.tensor([c[i] for c in conf_per])
        top_k_idx = confs.argsort(descending=True)[:k_agree]
        top_k_preds = torch.tensor([preds_per[j][i] for j in top_k_idx])
        if (top_k_preds == top_k_preds[0]).all():
            candidate = top_k_preds[0].item()
            if candidate != hybrid_preds[i].item():
                hybrid_preds[i] = candidate
                n_changed += 1

    report(f"topk_agree_k{k_agree}_(n={n_changed})", hybrid_preds)

print("\n" + "="*80)
print("STRATEGY 5: Geometric mean + CAFormer boost on disagreement")
print("Where CAFormers disagree with 6-model geo, boost CAFormer class prob")
print("="*80)
caf_geo = torch.exp(sum(torch.log(p + 1e-8) for p in all_caf) / len(all_caf))
caf_preds = caf_geo.argmax(1)

for boost in [0.1, 0.2, 0.3, 0.5]:
    disagree = (caf_preds != geo6_preds)
    boosted = geo6.clone()
    for i in range(n_samples):
        if disagree[i]:
            caf_class = caf_preds[i].item()
            boosted[i, caf_class] += boost * caf_geo[i, caf_class]
    preds = boosted.argmax(1)
    report(f"caf_boost_{boost}_(disagree={disagree.sum().item()})", preds)

print("\n" + "="*80)
print("STRATEGY 6: Architecture-level ensemble then combine")
print("Separate geo means per architecture, then combine")
print("="*80)
convnext_probs = [probs_6["v11_224"], probs_6["v18_384"], probs_6["v25_448"], probs_6["v38_512"]]
caformer_probs = [probs_6["vit_v9_224"]] + all_caf
effnet_probs = [probs_6["v24_384e"]]

geo_convnext = torch.exp(sum(torch.log(p+1e-8) for p in convnext_probs) / len(convnext_probs))
geo_caformer = torch.exp(sum(torch.log(p+1e-8) for p in caformer_probs) / len(caformer_probs))
geo_effnet = effnet_probs[0]

for w_conv, w_caf, w_eff in [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.6, 0.25, 0.15), (0.45, 0.35, 0.20)]:
    combined = torch.exp(
        w_conv * torch.log(geo_convnext + 1e-8) +
        w_caf * torch.log(geo_caformer + 1e-8) +
        w_eff * torch.log(geo_effnet + 1e-8)
    )
    preds = combined.argmax(1)
    report(f"arch_level_{w_conv}_{w_caf}_{w_eff}", preds)

print("\n" + "="*80)
print("STRATEGY 7: Selective replacement - only adopt CAFormer when it fixes")
print("Use 6-model base, replace with 8-model ONLY where 6-model top1 != top2")
print("(i.e., where 6-model is genuinely torn between two classes)")
print("="*80)
top2_6 = geo6.topk(2, dim=1)
margin = top2_6.values[:, 0] - top2_6.values[:, 1]

for margin_thresh in [0.05, 0.1, 0.15, 0.2]:
    close_call = margin < margin_thresh
    hybrid_preds = geo6_preds.clone()
    hybrid_preds[close_call] = geo8.argmax(1)[close_call]
    n_close = close_call.sum().item()
    report(f"margin_replace_t{margin_thresh}_(n={n_close})", hybrid_preds)

print("\n" + "="*80)
print("STRATEGY 8: Margin replace with arch-level CAFormer ensemble")
print("="*80)
geo_3caf = torch.exp(sum(torch.log(p+1e-8) for p in caformer_probs) / len(caformer_probs))
combo_6_plus_caf = torch.exp(
    0.7 * torch.log(geo6 + 1e-8) +
    0.3 * torch.log(geo_3caf + 1e-8)
)
combo_preds = combo_6_plus_caf.argmax(1)

for margin_thresh in [0.05, 0.1, 0.15]:
    close_call = margin < margin_thresh
    hybrid_preds = geo6_preds.clone()
    hybrid_preds[close_call] = combo_preds[close_call]
    n_close = close_call.sum().item()
    report(f"margin_arch_caf_t{margin_thresh}_(n={n_close})", hybrid_preds)

print("\n" + "="*80)
print("SUMMARY OF MOST CONSERVATIVE OPTIONS (< 30 changes)")
print("="*80)

print("\nDone!")
