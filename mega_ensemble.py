#!/usr/bin/env python3
"""
Mega-ensemble: Combine TTA probabilities from all model generations.
Weighted by CV accuracy. Generates final submission CSV.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

MODELS = {
    "v11":     {"dir": Path("/data/slr/checkpoints_v11"),     "cv": 90.35},
    "vit_v9":  {"dir": Path("/data/slr/checkpoints_vit_v9"),  "cv": 89.53},
    "v12":     {"dir": Path("/data/slr/checkpoints_v12"),      "cv": 87.78},
    "v13":     {"dir": Path("/data/slr/checkpoints_v13"),      "cv": None},
    "vit_v10": {"dir": Path("/data/slr/checkpoints_vit_v10"), "cv": None},
    "v12b":    {"dir": Path("/data/slr/checkpoints_v12b"),     "cv": None},
}

OUT_DIR = Path("/data/slr/mega_ensemble")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cv(model_dir):
    cv_path = model_dir / "cv_results.json"
    if cv_path.exists():
        with open(cv_path) as f:
            data = json.load(f)
        return data["mean"] * 100
    return None


def main():
    all_probs = {}
    all_cv = {}

    for name, cfg in MODELS.items():
        prob_path = cfg["dir"] / "eval_probs_tta.pt"
        if not prob_path.exists():
            print(f"  SKIP {name}: no TTA probs at {prob_path}")
            continue
        data = torch.load(prob_path, map_location="cpu", weights_only=True)
        all_probs[name] = data["probs"]
        keys = data["keys"]

        cv = cfg["cv"]
        if cv is None:
            cv = load_cv(cfg["dir"])
        if cv is not None:
            all_cv[name] = cv
            print(f"  {name:10s}  CV={cv:.2f}%  probs shape={data['probs'].shape}")
        else:
            all_cv[name] = 85.0
            print(f"  {name:10s}  CV=??  (using 85.0)  probs shape={data['probs'].shape}")

    if not all_probs:
        print("No models found!")
        return

    print(f"\nLoaded {len(all_probs)} models")

    # Compute CV-based weights (softmax of CV scores with temperature)
    names = list(all_probs.keys())
    cvs = np.array([all_cv[n] for n in names])
    temp = 2.0
    weights = np.exp((cvs - cvs.max()) / temp)
    weights /= weights.sum()

    print("\nWeights:")
    for n, w, cv in zip(names, weights, cvs):
        print(f"  {n:10s}  CV={cv:.2f}%  weight={w:.4f}")

    # Weighted average of probabilities
    ensemble_probs = torch.zeros_like(all_probs[names[0]])
    for n, w in zip(names, weights):
        ensemble_probs += w * all_probs[n]

    preds = ensemble_probs.argmax(dim=1).numpy()

    # Also compute individual model predictions for agreement analysis
    individual_preds = {n: all_probs[n].argmax(dim=1).numpy() for n in names}

    # Agreement analysis
    print(f"\nAgreement analysis ({len(preds)} samples):")
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            agree = (individual_preds[n1] == individual_preds[n2]).sum()
            print(f"  {n1:10s} vs {n2:10s}: {agree}/{len(preds)} "
                  f"({100*agree/len(preds):.1f}%)")

    # How many does the ensemble differ from each individual?
    print("\nEnsemble vs individual:")
    for n in names:
        diff = (preds != individual_preds[n]).sum()
        print(f"  Ensemble vs {n:10s}: {diff} differ "
              f"({100*diff/len(preds):.1f}%)")

    # Write submission
    sub_path = OUT_DIR / "submission_mega_ensemble.csv"
    with open(sub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "Pred"])
        for k, p in zip(keys, preds):
            sid = k.split("_", 1)[1]
            w.writerow([sid, int(p)])
    print(f"\nSubmission → {sub_path}  ({len(preds)} rows)")

    # Also save per-generation ensembles
    # Old generation: v11 + vit_v9 + v12
    old_names = [n for n in ["v11", "vit_v9", "v12"] if n in all_probs]
    if len(old_names) > 1:
        old_cvs = np.array([all_cv[n] for n in old_names])
        old_w = np.exp((old_cvs - old_cvs.max()) / temp)
        old_w /= old_w.sum()
        old_probs = sum(old_w[i] * all_probs[n]
                        for i, n in enumerate(old_names))
        old_preds = old_probs.argmax(dim=1).numpy()
        p1 = OUT_DIR / "submission_old_gen.csv"
        with open(p1, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "Pred"])
            for k, p in zip(keys, old_preds):
                w.writerow([k.split("_", 1)[1], int(p)])
        print(f"Old-gen ensemble → {p1}")

    # New generation: v13 + vit_v10 + v12b
    new_names = [n for n in ["v13", "vit_v10", "v12b"] if n in all_probs]
    if len(new_names) > 1:
        new_cvs = np.array([all_cv[n] for n in new_names])
        new_w = np.exp((new_cvs - new_cvs.max()) / temp)
        new_w /= new_w.sum()
        new_probs = sum(new_w[i] * all_probs[n]
                        for i, n in enumerate(new_names))
        new_preds = new_probs.argmax(dim=1).numpy()
        p2 = OUT_DIR / "submission_new_gen.csv"
        with open(p2, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "Pred"])
            for k, p in zip(keys, new_preds):
                w.writerow([k.split("_", 1)[1], int(p)])
        print(f"New-gen ensemble → {p2}")

    torch.save({"keys": keys, "probs": ensemble_probs},
               OUT_DIR / "mega_probs.pt")
    print(f"Saved mega probs → {OUT_DIR / 'mega_probs.pt'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
