#!/usr/bin/env python3
"""
OOF Stacking Ensemble:
1. Compute OOF predictions for all models (each fold's model predicts its val set)
2. Train a stacking model (logistic regression) on OOF predictions
3. Apply stacker to test predictions
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["OMP_NUM_THREADS"] = "4"
import torch, csv, json, logging
import torch.nn as nn, torch.nn.functional as F
import numpy as np, timm
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.ndimage import zoom

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("/data/slr/stacking_results/training.log"),
                              logging.StreamHandler()])
log = logging.getLogger(__name__)

OUT = Path("/data/slr/stacking_results")
OUT.mkdir(exist_ok=True)

N_CLASSES = 126
SEED = 42
CHANNEL_MEANS = np.array([-64.3217, -62.9119, -63.9254], dtype=np.float32).reshape(3,1,1)
CHANNEL_STDS = np.array([12.5721, 11.6620, 11.7259], dtype=np.float32).reshape(3,1,1)

class GeM(nn.Module):
    def __init__(self,p=3.0,eps=1e-6): super().__init__(); self.p=nn.Parameter(torch.ones(1)*p); self.eps=eps
    def forward(self,x): return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),1).pow(1.0/self.p)
class MultiSampleDropout(nn.Module):
    def __init__(self,inf,outf,n_samples=5,drop_rate=0.3):
        super().__init__(); self.dropouts=nn.ModuleList([nn.Dropout(drop_rate) for _ in range(n_samples)]); self.fc=nn.Linear(inf,outf)
    def forward(self,x): return torch.mean(torch.stack([self.fc(d(x)) for d in self.dropouts]),0) if self.training else self.fc(x)

class MultiScaleModel(nn.Module):
    def __init__(self, model_name, n_classes=126, pretrained=False,
                 drop_path_rate=0.0, proj_dim=512, drop_rate=0.3, ms_samples=5,
                 out_indices=(0,1,2,3)):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True,
                                          out_indices=out_indices, drop_path_rate=drop_path_rate)
        stage_dims = self.backbone.feature_info.channels()
        self.stage_gems = nn.ModuleList([GeM(p=3.0) for _ in stage_dims])
        self.stage_projs = nn.ModuleList([nn.Sequential(nn.Linear(d,proj_dim),nn.LayerNorm(proj_dim),nn.GELU()) for d in stage_dims])
        self.scale_attn = nn.Sequential(nn.Linear(proj_dim, 1))
        self.head = MultiSampleDropout(proj_dim, n_classes, n_samples=ms_samples, drop_rate=drop_rate)
    def forward(self, x):
        feats = self.backbone(x)
        projs = [proj(gem(f).flatten(1)) for gem,proj,f in zip(self.stage_gems,self.stage_projs,feats)]
        stacked = torch.stack(projs, dim=1)
        weights = F.softmax(self.scale_attn(stacked), dim=1)
        return self.head((stacked * weights).sum(dim=1))

class SimpleDS(Dataset):
    def __init__(self, keys, cache, img_size, max_frames=48):
        self.keys, self.cache, self.img_size, self.max_frames = keys, cache, img_size, max_frames
    def __len__(self): return len(self.keys)
    def __getitem__(self, idx):
        x = self.cache[self.keys[idx]].copy()
        x = (x - CHANNEL_MEANS) / CHANNEL_STDS
        t = x.shape[1]
        if t > self.max_frames:
            s = (t - self.max_frames) // 2; x = x[:, s:s+self.max_frames, :]
        elif t < self.max_frames:
            x = np.tile(x, (1, self.max_frames//t+1, 1))[:, :self.max_frames, :]
        t = torch.from_numpy(x)
        t = F.interpolate(t.unsqueeze(0), size=(self.img_size, self.img_size),
                         mode="bilinear", align_corners=False).squeeze(0)
        return t

def _tta(x):
    return [x,x.flip(2),x.roll(-4,2),x.roll(4,2),x[:,[1,2,0],:,:],x[:,[2,0,1],:,:],
            x+.05*torch.randn_like(x),x+.08*torch.randn_like(x),x*.9,x*1.1]

MODEL_CONFIGS = [
    {"name": "v11_224", "dir": "checkpoints_v11", "model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
     "img_size": 224, "proj_dim": 512, "drop_rate": 0.3},
    {"name": "vit_v9_224", "dir": "checkpoints_vit_v9", "model_name": "caformer_s18.sail_in22k_ft_in1k",
     "img_size": 224, "proj_dim": 384, "drop_rate": 0.3},
    {"name": "v18_384", "dir": "checkpoints_v18", "model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
     "img_size": 384, "proj_dim": 512, "drop_rate": 0.3},
    {"name": "v25_448", "dir": "checkpoints_v25", "model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
     "img_size": 448, "proj_dim": 512, "drop_rate": 0.3},
    {"name": "v24_384e", "dir": "checkpoints_v24", "model_name": "tf_efficientnetv2_s.in21k_ft_in1k",
     "img_size": 384, "proj_dim": 384, "drop_rate": 0.3, "out_indices": (0,1,2,3,4)},
    {"name": "v38_512", "dir": "checkpoints_v38", "model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
     "img_size": 512, "proj_dim": 512, "drop_rate": 0.3},
    {"name": "v43_320ca", "dir": "checkpoints_v43", "model_name": "caformer_s18.sail_in22k_ft_in1k",
     "img_size": 320, "proj_dim": 384, "drop_rate": 0.3},
]

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

def preload(samples, is_test=False):
    c = {}
    for item in tqdm(samples, desc="Pre-loading", leave=False):
        c[item[1]] = _load_one(item[0], item[1])
    return c

def write_sub(keys, preds, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "Pred"])
        for k, p in zip(keys, preds): w.writerow([k.split("_",1)[1], int(p)])

def main():
    dev = torch.device("cuda:0")
    log.info("GPU: %s", torch.cuda.get_device_name(0))

    train_samples = collect_train_samples("/data/slr/track2_")
    eval_samples = collect_eval_samples("/data/slr/track2_")
    keys = [s[1] for s in train_samples]
    labels = np.array([s[2] for s in train_samples])
    eval_keys = [s[1] for s in eval_samples]
    log.info("Train: %d | Eval: %d", len(keys), len(eval_keys))

    train_cache = preload(train_samples)
    eval_cache = preload(eval_samples, is_test=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(keys, labels))

    n_models = len(MODEL_CONFIGS)
    oof_probs = np.zeros((len(keys), N_CLASSES, n_models), dtype=np.float32)

    for mi, cfg in enumerate(MODEL_CONFIGS):
        log.info("="*60)
        log.info("Computing OOF for %s (%s @ %d)", cfg["name"], cfg["model_name"], cfg["img_size"])
        log.info("="*60)

        for fold_idx, (_, val_idx) in enumerate(folds):
            ckpt = Path(f"/data/slr/{cfg['dir']}/best_fold{fold_idx}.pt")
            if not ckpt.exists():
                log.warning("  Fold %d checkpoint missing: %s", fold_idx, ckpt)
                continue

            oi = cfg.get("out_indices", (0,1,2,3))
            log.info("  Creating model: %s proj_dim=%d out_indices=%s", cfg["model_name"], cfg["proj_dim"], oi)
            model = MultiScaleModel(
                model_name=cfg["model_name"],
                n_classes=N_CLASSES,
                pretrained=False,
                drop_path_rate=0.0,
                proj_dim=cfg["proj_dim"],
                drop_rate=cfg["drop_rate"],
                ms_samples=5,
                out_indices=oi,
            ).to(dev)
            sd = torch.load(ckpt, map_location=dev, weights_only=True)
            model.load_state_dict(sd["model_state_dict"])
            model.eval()

            val_keys = [keys[i] for i in val_idx]
            ds = SimpleDS(val_keys, train_cache, cfg["img_size"])
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

            all_probs = []
            with torch.no_grad():
                for x in tqdm(loader, desc=f"  Fold {fold_idx}", leave=False):
                    x = x.to(dev, non_blocking=True)
                    bp = torch.zeros(x.size(0), N_CLASSES, device=dev)
                    for v in _tta(x):
                        bp += F.softmax(model(v), 1)
                    all_probs.append((bp / 10).cpu().numpy())

            fold_probs = np.concatenate(all_probs)
            for local_i, global_i in enumerate(val_idx):
                oof_probs[global_i, :, mi] = fold_probs[local_i]

            fold_preds = fold_probs.argmax(1)
            fold_labels = labels[val_idx]
            fold_acc = (fold_preds == fold_labels).mean()
            log.info("  Fold %d: OOF acc = %.2f%%", fold_idx, 100*fold_acc)

            del model; torch.cuda.empty_cache()

        model_oof_preds = oof_probs[:, :, mi].argmax(1)
        model_oof_acc = (model_oof_preds == labels).mean()
        log.info("  %s OOF accuracy: %.2f%%", cfg["name"], 100*model_oof_acc)

    np.save(OUT / "oof_probs.npy", oof_probs)
    np.save(OUT / "labels.npy", labels)
    log.info("OOF predictions saved: %s", oof_probs.shape)

    log.info("="*60)
    log.info("STACKING ENSEMBLE")
    log.info("="*60)

    X_oof = oof_probs.reshape(len(keys), -1)
    log.info("Stacker input shape: %s", X_oof.shape)

    best6_names = ["v11_224", "vit_v9_224", "v18_384", "v25_448", "v24_384e", "v38_512"]
    best6_idx = [i for i, c in enumerate(MODEL_CONFIGS) if c["name"] in best6_names]

    for combo_name, model_indices in [
        ("all_7", list(range(n_models))),
        ("best_6", best6_idx),
        ("best_6_plus_v43", best6_idx + [6]),
    ]:
        idx = model_indices
        X = oof_probs[:, :, idx].reshape(len(keys), -1)
        log.info("\n--- %s (%d models, %d features) ---", combo_name, len(idx), X.shape[1])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        geo_probs = np.exp(np.mean(np.log(oof_probs[:, :, idx] + 1e-8), axis=2))
        geo_preds = geo_probs.argmax(1)
        geo_acc = (geo_preds == labels).mean()
        log.info("  Geometric mean OOF acc: %.2f%%", 100*geo_acc)

        for C in [0.01, 0.1, 1.0, 10.0]:
            lr = LogisticRegression(C=C, max_iter=1000, solver='lbfgs',
                                   multi_class='multinomial', random_state=SEED, n_jobs=-1)
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(lr, X_scaled, labels, cv=5, scoring='accuracy', n_jobs=-1)
            log.info("  LR C=%.2f: %.2f%% ± %.2f%% (5-fold inner CV)", C, 100*cv_scores.mean(), 100*cv_scores.std())

        best_C = 1.0
        lr_final = LogisticRegression(C=best_C, max_iter=1000, solver='lbfgs',
                                      multi_class='multinomial', random_state=SEED, n_jobs=-1)
        lr_final.fit(X_scaled, labels)
        train_acc = lr_final.score(X_scaled, labels)
        log.info("  Final LR (C=%.1f) train acc: %.2f%%", best_C, 100*train_acc)

        eval_probs_list = []
        for mi in idx:
            cfg = MODEL_CONFIGS[mi]
            data = torch.load(f"/data/slr/{cfg['dir']}/eval_probs_tta.pt",
                            map_location='cpu', weights_only=True)
            eval_probs_list.append(data['probs'].numpy())
        eval_probs_stack = np.stack(eval_probs_list, axis=2)
        X_eval = eval_probs_stack.reshape(len(eval_keys), -1)
        X_eval_scaled = scaler.transform(X_eval)

        stacked_preds = lr_final.predict(X_eval_scaled)
        write_sub(eval_keys, stacked_preds, OUT / f"stacked_{combo_name}_C{best_C}.csv")

        stacked_proba = lr_final.predict_proba(X_eval_scaled)
        geo_eval = np.exp(np.mean(np.log(eval_probs_stack + 1e-8), axis=2))
        geo_preds_eval = geo_eval.argmax(1)

        diff = (stacked_preds != geo_preds_eval).sum()
        log.info("  Stacked vs Geo: %d changes on test set", diff)

    log.info("\nDone! All files in %s", OUT)

if __name__ == "__main__":
    main()
