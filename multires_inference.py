#!/usr/bin/env python3
"""
Multi-Resolution Inference: evaluate existing models at multiple resolutions.
ConvNeXtV2 uses LayerNorm (not BatchNorm) → resolution-independent weights.
GeM pooling is also resolution-independent. So v11 trained at 224 can be
evaluated at 256, 288, 320, 384 without any degradation from architecture.
This gives us multi-resolution predictions with ZERO training cost.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import torch; torch.set_num_threads(4)
import csv, json, numpy as np, timm
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

RESOLUTIONS = [224, 256, 288, 320, 384]
MAX_FRAMES = 48

CFG_BASE = {
    "data_dir": "/data/slr/track2_",
    "n_classes": 126, "n_folds": 5,
    "channel_means": [-64.3217, -62.9119, -63.9254],
    "channel_stds": [12.5721, 11.6620, 11.7259],
}

def collect_eval_samples(dd):
    s = []
    for sp in ("val", "test"):
        sd = os.path.join(dd, sp)
        if not os.path.isdir(sd): continue
        for sn in sorted(os.listdir(sd)):
            sp2 = os.path.join(sd, sn)
            if os.path.isdir(sp2): s.append((sp2, sn))
    return s

def _load_one(sp, sn):
    return np.stack([np.load(os.path.join(sp, f"{sn}_RTM{r}.npy")) for r in (1,2,3)], 0).astype(np.float32)

_MEANS = np.array(CFG_BASE["channel_means"], dtype=np.float32).reshape(3,1,1)
_STDS = np.array(CFG_BASE["channel_stds"], dtype=np.float32).reshape(3,1,1)

class EvalDataset(Dataset):
    def __init__(self, keys, cache, img_size=224):
        self.keys, self.cache, self.img_size = keys, cache, img_size
    def __len__(self): return len(self.keys)
    def __getitem__(self, idx):
        x = self.cache[self.keys[idx]].copy()
        x = (x - _MEANS) / _STDS
        t = x.shape[1]
        if t > MAX_FRAMES:
            s = (t - MAX_FRAMES) // 2
            x = x[:, s:s+MAX_FRAMES, :]
        elif t < MAX_FRAMES:
            x = np.tile(x, (1, MAX_FRAMES//t+1, 1))[:, :MAX_FRAMES, :]
        t = torch.from_numpy(x)
        sz = (self.img_size, self.img_size)
        t = F.interpolate(t.unsqueeze(0), size=sz, mode="bilinear", align_corners=False).squeeze(0)
        return t, 0, torch.zeros(CFG_BASE["n_classes"])

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6): super().__init__(); self.p=nn.Parameter(torch.ones(1)*p); self.eps=eps
    def forward(self, x): return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0/self.p)

class MultiSampleDropout(nn.Module):
    def __init__(self, inf, outf, n_samples=5, drop_rate=0.3):
        super().__init__(); self.dropouts=nn.ModuleList([nn.Dropout(drop_rate) for _ in range(n_samples)]); self.fc=nn.Linear(inf, outf)
    def forward(self, x): return torch.mean(torch.stack([self.fc(d(x)) for d in self.dropouts]), 0) if self.training else self.fc(x)

class MultiScaleConvNeXtV2(nn.Module):
    def __init__(self, model_name, n_classes=126, pretrained=True,
                 drop_path_rate=0.2, proj_dim=512, drop_rate=0.3, ms_samples=5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True,
                                          out_indices=(0,1,2,3), drop_path_rate=drop_path_rate)
        stage_dims = self.backbone.feature_info.channels()
        self.stage_gems = nn.ModuleList([GeM(p=3.0) for _ in stage_dims])
        self.stage_projs = nn.ModuleList([nn.Sequential(nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU()) for d in stage_dims])
        self.scale_attn = nn.Sequential(nn.Linear(proj_dim, 1))
        self.head = MultiSampleDropout(proj_dim, n_classes, n_samples=ms_samples, drop_rate=drop_rate)
    def forward(self, x):
        feats = self.backbone(x)
        projs = [proj(gem(f).flatten(1)) for gem, proj, f in zip(self.stage_gems, self.stage_projs, feats)]
        stacked = torch.stack(projs, dim=1)
        weights = F.softmax(self.scale_attn(stacked), dim=1)
        return self.head((stacked * weights).sum(dim=1))

class MultiScaleCAFormer(nn.Module):
    def __init__(self, model_name, n_classes=126, pretrained=True,
                 drop_path_rate=0.15, proj_dim=384, drop_rate=0.3, ms_samples=5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True,
                                          out_indices=(0,1,2,3), drop_path_rate=drop_path_rate)
        stage_dims = self.backbone.feature_info.channels()
        self.stage_gems = nn.ModuleList([GeM(p=3.0) for _ in stage_dims])
        self.stage_projs = nn.ModuleList([nn.Sequential(nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU()) for d in stage_dims])
        self.scale_attn = nn.Sequential(nn.Linear(proj_dim, 1))
        self.head = MultiSampleDropout(proj_dim, n_classes, n_samples=ms_samples, drop_rate=drop_rate)
    def forward(self, x):
        feats = self.backbone(x)
        projs = [proj(gem(f).flatten(1)) for gem, proj, f in zip(self.stage_gems, self.stage_projs, feats)]
        stacked = torch.stack(projs, dim=1)
        weights = F.softmax(self.scale_attn(stacked), dim=1)
        return self.head((stacked * weights).sum(dim=1))

def _tta(x):
    return [x, x.flip(2), x.roll(-8,2), x.roll(8,2), x[:,[1,2,0],:,:], x[:,[2,0,1],:,:],
            x+.05*torch.randn_like(x), x+.08*torch.randn_like(x), x*.9, x*1.1]

@torch.no_grad()
def predict_at_res(models, keys, cache, res, dev, use_tta=True):
    ds = EvalDataset(keys, cache, img_size=res)
    loader = DataLoader(ds, batch_size=48, shuffle=False, num_workers=4, pin_memory=True)
    for m in models: m.eval()
    all_probs = []
    for x, _, _ in tqdm(loader, desc=f"  res={res}", leave=False):
        x = x.to(dev, non_blocking=True)
        bp = torch.zeros(x.size(0), CFG_BASE["n_classes"], device=dev)
        views = _tta(x) if use_tta else [x]
        for v in views:
            for m in models: bp += F.softmax(m(v), 1)
        all_probs.append((bp / (len(views) * len(models))).cpu())
    return torch.cat(all_probs)

def write_sub(keys, preds, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "Pred"])
        for k, p in zip(keys, preds): w.writerow([k.split("_",1)[1], int(p)])

def main():
    dev = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    out = Path("/data/slr/submissions_multires_tta")
    out.mkdir(parents=True, exist_ok=True)

    eval_samples = collect_eval_samples(CFG_BASE["data_dir"])
    eval_keys = [s[1] for s in eval_samples]
    print(f"Eval samples: {len(eval_keys)}")

    print("Pre-loading eval data...")
    cache = {}
    for sp, sn in tqdm(eval_samples, desc="Loading"):
        cache[sn] = _load_one(sp, sn)

    # Load v11 models (5 folds)
    v11_dir = Path("/data/slr/checkpoints_v11")
    v11_models = []
    for f in range(5):
        m = MultiScaleConvNeXtV2("convnextv2_tiny.fcmae_ft_in22k_in1k", 126, pretrained=False,
                                  drop_path_rate=0.0, proj_dim=512, drop_rate=0.3, ms_samples=5).to(dev)
        m.load_state_dict(torch.load(v11_dir / f"best_fold{f}.pt", map_location=dev, weights_only=True)["model_state_dict"])
        v11_models.append(m)
    print(f"Loaded {len(v11_models)} v11 models")

    # Multi-resolution inference for v11
    v11_res_probs = {}
    for res in RESOLUTIONS:
        print(f"v11 @ {res}x{res}...")
        v11_res_probs[res] = predict_at_res(v11_models, eval_keys, cache, res, dev, use_tta=True)
        print(f"  Shape: {v11_res_probs[res].shape}")

    # Save individual resolution predictions
    for res in RESOLUTIONS:
        preds = v11_res_probs[res].argmax(1).numpy()
        write_sub(eval_keys, preds, out / f"v11_at_{res}.csv")
        print(f"  v11@{res}: {out / f'v11_at_{res}.csv'}")

    # Multi-resolution average for v11 alone
    avg_probs = sum(v11_res_probs[r] for r in RESOLUTIONS) / len(RESOLUTIONS)
    write_sub(eval_keys, avg_probs.argmax(1).numpy(), out / "v11_multires_avg.csv")
    print(f"v11 multi-res avg: {out / 'v11_multires_avg.csv'}")

    # Geometric mean across resolutions for v11
    geo = torch.ones_like(v11_res_probs[224])
    for r in RESOLUTIONS:
        geo *= v11_res_probs[r]
    geo = geo.pow(1.0 / len(RESOLUTIONS))
    geo = geo / geo.sum(1, keepdim=True)
    write_sub(eval_keys, geo.argmax(1).numpy(), out / "v11_multires_geometric.csv")

    del v11_models; torch.cuda.empty_cache()

    # Load vit_v9 models (5 folds)
    vit_dir = Path("/data/slr/checkpoints_vit_v9")
    vit_models = []
    for f in range(5):
        m = MultiScaleCAFormer("caformer_s18.sail_in22k_ft_in1k", 126, pretrained=False,
                                drop_path_rate=0.0, proj_dim=384, drop_rate=0.3, ms_samples=5).to(dev)
        m.load_state_dict(torch.load(vit_dir / f"best_fold{f}.pt", map_location=dev, weights_only=True)["model_state_dict"])
        vit_models.append(m)
    print(f"Loaded {len(vit_models)} vit_v9 models")

    # Multi-res for vit_v9 (skip 384 — CAFormer is slow there)
    vit_resolutions = [224, 256, 288]
    vit_res_probs = {}
    for res in vit_resolutions:
        print(f"vit_v9 @ {res}x{res}...")
        vit_res_probs[res] = predict_at_res(vit_models, eval_keys, cache, res, dev, use_tta=True)

    del vit_models; torch.cuda.empty_cache()

    # Load v18 probs (already at 384)
    v18_data = torch.load("/data/slr/checkpoints_v18/eval_probs_tta.pt", map_location="cpu", weights_only=False)
    v18_probs = v18_data["probs"]

    # === MEGA Multi-Resolution Ensembles ===
    print("\n=== Multi-Resolution Mega Ensembles ===")

    # 1. v11 multi-res + vit_v9@224 + v18@384
    all_probs = [v11_res_probs[r] for r in RESOLUTIONS] + [vit_res_probs[224], v18_probs]
    mega = sum(all_probs) / len(all_probs)
    write_sub(eval_keys, mega.argmax(1).numpy(), out / "mega_multires_arithmetic.csv")
    print(f"1. Mega multi-res arithmetic: {len(all_probs)} views")

    # 2. Geometric mean version
    geo_mega = torch.ones_like(all_probs[0])
    for p in all_probs: geo_mega *= p
    geo_mega = geo_mega.pow(1.0 / len(all_probs))
    geo_mega = geo_mega / geo_mega.sum(1, keepdim=True)
    write_sub(eval_keys, geo_mega.argmax(1).numpy(), out / "mega_multires_geometric.csv")
    print(f"2. Mega multi-res geometric: {len(all_probs)} views")

    # 3. v11 multi-res + vit_v9 multi-res + v18@384
    all_probs_v2 = [v11_res_probs[r] for r in RESOLUTIONS] + [vit_res_probs[r] for r in vit_resolutions] + [v18_probs]
    mega_v2 = sum(all_probs_v2) / len(all_probs_v2)
    write_sub(eval_keys, mega_v2.argmax(1).numpy(), out / "mega_multires_v2_arithmetic.csv")
    print(f"3. Full multi-res v2 arithmetic: {len(all_probs_v2)} views")

    geo_v2 = torch.ones_like(all_probs_v2[0])
    for p in all_probs_v2: geo_v2 *= p
    geo_v2 = geo_v2.pow(1.0 / len(all_probs_v2))
    geo_v2 = geo_v2 / geo_v2.sum(1, keepdim=True)
    write_sub(eval_keys, geo_v2.argmax(1).numpy(), out / "mega_multires_v2_geometric.csv")
    print(f"4. Full multi-res v2 geometric: {len(all_probs_v2)} views")

    # 5. Just v11@224 + v11@384 + v18@384 + vit_v9@224 (trained vs inferred at high-res)
    combo = (v11_res_probs[224] + v11_res_probs[384] + v18_probs + vit_res_probs[224]) / 4
    write_sub(eval_keys, combo.argmax(1).numpy(), out / "v11_224_384_v18_384_vitv9_224.csv")
    print("5. v11@{224,384} + v18@384 + vit_v9@224")

    # 6. Best triple (v11+vit_v9+v18) + v11 at extra resolutions
    base_triple = (v11_res_probs[224] + vit_res_probs[224] + v18_probs)
    extra = sum(v11_res_probs[r] for r in [256, 288, 320])
    enhanced = (base_triple + extra) / 6
    write_sub(eval_keys, enhanced.argmax(1).numpy(), out / "triple_plus_v11_multires.csv")
    print("6. Base triple + v11@{256,288,320}")

    geo_enhanced = (v11_res_probs[224] * vit_res_probs[224] * v18_probs *
                    v11_res_probs[256] * v11_res_probs[288] * v11_res_probs[320]).pow(1/6)
    geo_enhanced = geo_enhanced / geo_enhanced.sum(1, keepdim=True)
    write_sub(eval_keys, geo_enhanced.argmax(1).numpy(), out / "triple_plus_v11_multires_geometric.csv")
    print("7. Geometric version of #6")

    # Compare with old best
    old = {}
    with open("/data/slr/ensemble_v11_vit9_submission.csv") as f:
        r = csv.reader(f); next(r)
        for row in r: old[int(row[0])] = int(row[1])
    best89 = {}
    with open("/data/slr/submissions_multiresolution/ensemble_v11_vitv9_v18_geometric.csv") as f:
        r = csv.reader(f); next(r)
        for row in r: best89[int(row[0])] = int(row[1])

    print("\n=== Comparison vs 89.72% best ===")
    for name in ["mega_multires_arithmetic", "mega_multires_geometric",
                  "mega_multires_v2_arithmetic", "mega_multires_v2_geometric",
                  "triple_plus_v11_multires", "triple_plus_v11_multires_geometric",
                  "v11_multires_avg", "v11_multires_geometric"]:
        cur = {}
        with open(out / f"{name}.csv") as f:
            r = csv.reader(f); next(r)
            for row in r: cur[int(row[0])] = int(row[1])
        common = set(best89) & set(cur)
        changed = sum(1 for k in common if best89[k] != cur[k])
        print(f"  {name}: {changed}/{len(common)} changed vs 89.72% ({100*changed/len(common):.1f}%)")

    print("\nDone!")

if __name__ == "__main__":
    main()
