#!/usr/bin/env python3
"""
Compute Out-Of-Fold (OOF) predictions for all models.
For each fold, loads the checkpoint trained WITHOUT that fold's data,
and predicts on exactly those held-out samples.
Result: every training sample gets a prediction from a model that never saw it.
This is the only unbiased accuracy estimate we can compute.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, timm, json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = "/data/slr/track2_"
N_CLASSES = 126
SEED = 42
N_FOLDS = 5
_MEANS = np.array([-64.3217, -62.9119, -63.9254], dtype=np.float32).reshape(3,1,1)
_STDS = np.array([12.5721, 11.6620, 11.7259], dtype=np.float32).reshape(3,1,1)

def collect_train_samples(dd):
    td = os.path.join(dd, "train"); s = []
    for cf in sorted(os.listdir(td)):
        cp = os.path.join(td, cf)
        if not os.path.isdir(cp): continue
        ci = int(cf.split("_",1)[0])
        for sn in sorted(os.listdir(cp)):
            sp = os.path.join(cp, sn)
            if os.path.isdir(sp): s.append((sp, sn, ci))
    return s

def _load_one(sp, sn):
    return np.stack([np.load(os.path.join(sp, f"{sn}_RTM{r}.npy")) for r in (1,2,3)], 0).astype(np.float32)

def preload(samples):
    c = {}
    for item in tqdm(samples, desc="Pre-loading", leave=False):
        c[item[1]] = _load_one(item[0], item[1])
    return c

class EvalDataset(Dataset):
    def __init__(self, keys, cache, img_size=224, max_frames=48):
        self.keys, self.cache = keys, cache
        self.img_size, self.max_frames = img_size, max_frames
    def __len__(self): return len(self.keys)
    def __getitem__(self, idx):
        x = self.cache[self.keys[idx]].copy()
        x = (x - _MEANS) / _STDS
        t = x.shape[1]
        if t > self.max_frames:
            s = (t - self.max_frames) // 2
            x = x[:, s:s+self.max_frames, :]
        elif t < self.max_frames:
            x = np.tile(x, (1, self.max_frames//t+1, 1))[:, :self.max_frames, :]
        t = torch.from_numpy(x)
        sz = (self.img_size, self.img_size)
        t = F.interpolate(t.unsqueeze(0), size=sz, mode="bilinear", align_corners=False).squeeze(0)
        return t

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6): super().__init__(); self.p=nn.Parameter(torch.ones(1)*p); self.eps=eps
    def forward(self, x): return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0/self.p)
class MultiSampleDropout(nn.Module):
    def __init__(self, inf, outf, n_samples=5, drop_rate=0.3):
        super().__init__(); self.dropouts=nn.ModuleList([nn.Dropout(drop_rate) for _ in range(n_samples)]); self.fc=nn.Linear(inf, outf)
    def forward(self, x): return torch.mean(torch.stack([self.fc(d(x)) for d in self.dropouts]), 0) if self.training else self.fc(x)

class MultiScaleConvNeXtV2(nn.Module):
    def __init__(self, model_name, n_classes=126, pretrained=False, drop_path_rate=0.0, proj_dim=512, drop_rate=0.3, ms_samples=5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(0,1,2,3), drop_path_rate=drop_path_rate)
        sd = self.backbone.feature_info.channels()
        self.stage_gems = nn.ModuleList([GeM() for _ in sd])
        self.stage_projs = nn.ModuleList([nn.Sequential(nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU()) for d in sd])
        self.scale_attn = nn.Sequential(nn.Linear(proj_dim, 1))
        self.head = MultiSampleDropout(proj_dim, n_classes, n_samples=ms_samples, drop_rate=drop_rate)
    def forward(self, x):
        feats = self.backbone(x)
        projs = [proj(gem(f).flatten(1)) for gem,proj,f in zip(self.stage_gems, self.stage_projs, feats)]
        stacked = torch.stack(projs, dim=1)
        return self.head((stacked * F.softmax(self.scale_attn(stacked), dim=1)).sum(dim=1))

class MultiScaleCAFormer(nn.Module):
    def __init__(self, model_name, n_classes=126, pretrained=False, drop_path_rate=0.0, proj_dim=384, drop_rate=0.3, ms_samples=5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(0,1,2,3), drop_path_rate=drop_path_rate)
        sd = self.backbone.feature_info.channels()
        self.stage_gems = nn.ModuleList([GeM() for _ in sd])
        self.stage_projs = nn.ModuleList([nn.Sequential(nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU()) for d in sd])
        self.scale_attn = nn.Sequential(nn.Linear(proj_dim, 1))
        self.head = MultiSampleDropout(proj_dim, n_classes, n_samples=ms_samples, drop_rate=drop_rate)
    def forward(self, x):
        feats = self.backbone(x)
        projs = [proj(gem(f).flatten(1)) for gem,proj,f in zip(self.stage_gems, self.stage_projs, feats)]
        stacked = torch.stack(projs, dim=1)
        return self.head((stacked * F.softmax(self.scale_attn(stacked), dim=1)).sum(dim=1))

class MultiScaleEfficientNetV2(nn.Module):
    def __init__(self, model_name, n_classes=126, pretrained=False, drop_path_rate=0.0, proj_dim=384, drop_rate=0.3, ms_samples=5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(0,1,2,3,4), drop_path_rate=drop_path_rate)
        sd = self.backbone.feature_info.channels()
        self.stage_gems = nn.ModuleList([GeM() for _ in sd])
        self.stage_projs = nn.ModuleList([nn.Sequential(nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU()) for d in sd])
        self.scale_attn = nn.Sequential(nn.Linear(proj_dim, 1))
        self.head = MultiSampleDropout(proj_dim, n_classes, n_samples=ms_samples, drop_rate=drop_rate)
    def forward(self, x):
        feats = self.backbone(x)
        projs = [proj(gem(f).flatten(1)) for gem,proj,f in zip(self.stage_gems, self.stage_projs, feats)]
        stacked = torch.stack(projs, dim=1)
        return self.head((stacked * F.softmax(self.scale_attn(stacked), dim=1)).sum(dim=1))

MODELS = {
    "v11_224": {
        "ckpt_dir": "checkpoints_v11", "cls": MultiScaleConvNeXtV2,
        "model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "kwargs": {"proj_dim": 512, "drop_rate": 0.3, "ms_samples": 5},
        "img_size": 224, "seed": 42,
    },
    "v14_224": {
        "ckpt_dir": "checkpoints_v14", "cls": MultiScaleConvNeXtV2,
        "model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "kwargs": {"proj_dim": 512, "drop_rate": 0.3, "ms_samples": 5},
        "img_size": 224, "seed": 42,
    },
    "v18_384": {
        "ckpt_dir": "checkpoints_v18", "cls": MultiScaleConvNeXtV2,
        "model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "kwargs": {"proj_dim": 512, "drop_rate": 0.3, "ms_samples": 5},
        "img_size": 384, "seed": 42,
    },
    "v19_288": {
        "ckpt_dir": "checkpoints_v19", "cls": MultiScaleConvNeXtV2,
        "model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "kwargs": {"proj_dim": 512, "drop_rate": 0.3, "ms_samples": 5},
        "img_size": 288, "seed": 42,
    },
    "v20_320": {
        "ckpt_dir": "checkpoints_v20", "cls": MultiScaleConvNeXtV2,
        "model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "kwargs": {"proj_dim": 512, "drop_rate": 0.3, "ms_samples": 5},
        "img_size": 320, "seed": 42,
    },
    "v25_448": {
        "ckpt_dir": "checkpoints_v25", "cls": MultiScaleConvNeXtV2,
        "model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "kwargs": {"proj_dim": 512, "drop_rate": 0.3, "ms_samples": 5},
        "img_size": 448, "seed": 42,
    },
    "vit_v9_224": {
        "ckpt_dir": "checkpoints_vit_v9", "cls": MultiScaleCAFormer,
        "model_name": "caformer_s18.sail_in22k_ft_in1k",
        "kwargs": {"proj_dim": 384, "drop_rate": 0.3, "ms_samples": 5},
        "img_size": 224, "seed": 42,
    },
    "vit_v14_384": {
        "ckpt_dir": "checkpoints_vit_v14", "cls": MultiScaleCAFormer,
        "model_name": "caformer_s18.sail_in22k_ft_in1k",
        "kwargs": {"proj_dim": 384, "drop_rate": 0.3, "ms_samples": 5},
        "img_size": 384, "seed": 42,
    },
    "v24_384": {
        "ckpt_dir": "checkpoints_v24", "cls": MultiScaleEfficientNetV2,
        "model_name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "kwargs": {"proj_dim": 384, "drop_rate": 0.3, "ms_samples": 5},
        "img_size": 384, "seed": 42,
    },
}

@torch.no_grad()
def predict_fold(model, keys, cache, img_size, dev, batch_size=64):
    ds = EvalDataset(keys, cache, img_size=img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    all_probs = []
    for x in loader:
        x = x.to(dev, non_blocking=True)
        logits = model(x)
        all_probs.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(all_probs)

def main():
    dev = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_samples = collect_train_samples(DATA_DIR)
    keys = [s[1] for s in train_samples]
    labels = np.array([s[2] for s in train_samples])
    n = len(keys)
    print(f"Training samples: {n}")
    cache = preload(train_samples)

    results = {}

    for mname, mcfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Computing OOF for: {mname}")
        print(f"{'='*60}")

        ckpt_dir = Path(f"/data/slr/{mcfg['ckpt_dir']}")
        seed = mcfg["seed"]
        img_size = mcfg["img_size"]

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_probs = torch.zeros(n, N_CLASSES)
        oof_mask = torch.zeros(n, dtype=torch.bool)

        for fold, (tri, vai) in enumerate(skf.split(keys, labels)):
            cp = ckpt_dir / f"best_fold{fold}.pt"
            if not cp.exists():
                print(f"  Fold {fold}: checkpoint missing, skip")
                continue

            model = mcfg["cls"](mcfg["model_name"], n_classes=N_CLASSES, pretrained=False,
                                **mcfg["kwargs"]).to(dev)
            ck = torch.load(cp, map_location=dev, weights_only=True)
            model.load_state_dict(ck["model_state_dict"])

            val_keys = [keys[i] for i in vai]
            probs = predict_fold(model, val_keys, cache, img_size, dev)
            oof_probs[vai] = probs
            oof_mask[vai] = True

            preds = probs.argmax(1).numpy()
            val_labels = labels[vai]
            acc = (preds == val_labels).mean()
            print(f"  Fold {fold}: val acc = {100*acc:.2f}%")
            del model; torch.cuda.empty_cache()

        oof_preds = oof_probs.argmax(1).numpy()
        oof_acc = (oof_preds[oof_mask.numpy()] == labels[oof_mask.numpy()]).mean()
        print(f"  OOF accuracy: {100*oof_acc:.2f}%")
        results[mname] = {"oof_probs": oof_probs, "oof_acc": oof_acc, "seed": seed}

    # Save OOF probs
    torch.save({"keys": keys, "labels": labels, "results": results},
               "/data/slr/oof_predictions.pt")

    # Ensemble OOF accuracy (only for models with same seed=42)
    print(f"\n{'='*60}")
    print("ENSEMBLE OOF ACCURACY (true held-out)")
    print(f"{'='*60}")

    seed42_models = {k: v for k, v in results.items() if v["seed"] == 42}

    def geo_oof(names):
        log_p = sum((results[n]["oof_probs"] + 1e-8).log() for n in names) / len(names)
        p = log_p.exp()
        preds = p.argmax(1).numpy()
        acc = (preds == labels).mean()
        return acc

    combos = [
        ("v11+vit_v9+v18 (baseline)", ["v11_224", "vit_v9_224", "v18_384"]),
        ("v11+vit_v9+v25", ["v11_224", "vit_v9_224", "v25_448"]),
        ("v11+vit_v9+v18+v25", ["v11_224", "vit_v9_224", "v18_384", "v25_448"]),
        ("v11+vit_v9+v18+v25+v24", ["v11_224", "vit_v9_224", "v18_384", "v25_448", "v24_384"]),
        ("v11+vit_v9+v25+v24", ["v11_224", "vit_v9_224", "v25_448", "v24_384"]),
        ("all 9 models", list(seed42_models.keys())),
    ]

    for label, names in combos:
        valid = [n for n in names if n in results]
        if len(valid) == len(names):
            acc = geo_oof(valid)
            print(f"  {label}: {100*acc:.2f}%")
        else:
            print(f"  {label}: SKIP (missing models)")

    # Individual model summary
    print(f"\n{'='*60}")
    print("INDIVIDUAL MODEL OOF ACCURACY (sorted)")
    print(f"{'='*60}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["oof_acc"]):
        print(f"  {name}: {100*r['oof_acc']:.2f}%")

if __name__ == "__main__":
    main()
