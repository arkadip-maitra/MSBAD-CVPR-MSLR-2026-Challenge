#!/usr/bin/env python3
"""
v23: ConvNeXtV2-BASE (88M params, 3× Tiny) @ 224×224 + SWA
Bigger model = more capacity to solve hard samples that all Tiny models get wrong.
No born-again (different arch size). ImageNet pretrained + KD from v11+vit_v9.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_MAX_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
import torch; torch.set_num_threads(4)
import csv, json, logging, time
from pathlib import Path
import numpy as np, timm, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomErasing
from tqdm import tqdm

OUT_DIR = Path("/data/slr/checkpoints_v23")
V11_DIR = Path("/data/slr/checkpoints_v11")
VIT_V9_DIR = Path("/data/slr/checkpoints_vit_v9")
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(OUT_DIR/"training.log"), logging.StreamHandler()])
log = logging.getLogger(__name__)

CFG = {
    "data_dir": "/data/slr/track2_", "output_dir": str(OUT_DIR),
    "max_frames": 48, "img_size": 224, "n_classes": 126,
    "teacher_img_size": 224, "teacher_max_frames": 48,
    "channel_means": [-64.3217, -62.9119, -63.9254],
    "channel_stds": [12.5721, 11.6620, 11.7259],
    "n_folds": 5, "batch_size": 32, "num_workers": 4,
    "stage1_epochs": 8, "stage1_lr": 3e-4,
    "stage2_epochs": 100, "stage2_lr": 3e-5,
    "swa_epochs": 15, "swa_lr": 3e-6,
    "weight_decay": 0.05, "warmup_pct": 0.08,
    "label_smoothing": 0.1, "early_stopping_patience": 35,
    "grad_clip_norm": 1.0, "llrd_decay": 0.80,
    "kd_alpha": 0.5, "kd_tau": 4.0,
    "mixup_alpha": 0.4, "cutmix_alpha": 1.0, "cutmix_prob": 0.5,
    "time_mask_max": 8, "range_mask_max": 30, "noise_std": 0.1,
    "time_warp_lo": 0.8, "time_warp_hi": 1.2,
    "time_reverse_prob": 0.5, "channel_shuffle_prob": 0.2, "channel_drop_prob": 0.1,
    "random_erase_prob": 0.20, "circ_shift_prob": 0.3, "circ_shift_max": 8,
    "amp_scale_prob": 0.5, "amp_scale_lo": 0.8, "amp_scale_hi": 1.2,
    "model_name": "convnextv2_base.fcmae_ft_in22k_in1k",
    "drop_path_rate": 0.3, "proj_dim": 512, "drop_rate": 0.4,
    "ms_dropout_samples": 5, "seed": 42,
}

def collect_train_samples(dd):
    td=os.path.join(dd,"train"); s,m=[],{}
    for cf in sorted(os.listdir(td)):
        cp=os.path.join(td,cf)
        if not os.path.isdir(cp): continue
        ci=int(cf.split("_",1)[0]); m[ci]=cf
        for sn in sorted(os.listdir(cp)):
            sp=os.path.join(cp,sn)
            if os.path.isdir(sp): s.append((sp,sn,ci))
    return s,m
def collect_eval_samples(dd):
    s=[]
    for sp in ("val","test"):
        sd=os.path.join(dd,sp)
        if not os.path.isdir(sd): continue
        for sn in sorted(os.listdir(sd)):
            sp2=os.path.join(sd,sn)
            if os.path.isdir(sp2): s.append((sp2,sn))
    return s
def _load_one(sp,sn):
    return np.stack([np.load(os.path.join(sp,f"{sn}_RTM{r}.npy")) for r in (1,2,3)],0).astype(np.float32)
def preload(samples, is_test=False):
    c={}
    for item in tqdm(samples, desc="Pre-loading", leave=False):
        c[item[1] if is_test else item[1]]=_load_one(item[0],item[1])
    return c

_MEANS=np.array(CFG["channel_means"],dtype=np.float32).reshape(3,1,1)
_STDS=np.array(CFG["channel_stds"],dtype=np.float32).reshape(3,1,1)
_ERASER=RandomErasing(p=CFG["random_erase_prob"],scale=(0.02,0.2),ratio=(0.3,3.3),value=0)

class RadarDataset(Dataset):
    def __init__(self, keys, labels, cache, augment=False, soft_labels=None,
                 img_size=224, max_frames=48):
        self.keys, self.labels, self.cache = keys, labels, cache
        self.augment, self.soft_labels = augment, soft_labels
        self.img_size, self.max_frames = img_size, max_frames

    def __len__(self): return len(self.keys)
    @staticmethod
    def _pad_or_crop(x, max_t, random_crop=False):
        t=x.shape[1]
        if t>max_t:
            s=np.random.randint(0,t-max_t+1) if random_crop else (t-max_t)//2
            x=x[:,s:s+max_t,:]
        elif t<max_t:
            x=np.tile(x,(1,max_t//t+1,1))[:,:max_t,:]
        return x
    def _apply_aug(self, x):
        mf = self.max_frames
        if np.random.random()<CFG["time_reverse_prob"]: x=x[:,::-1,:].copy()
        if np.random.random()<CFG["channel_shuffle_prob"]: x=x[np.random.permutation(3)]
        if np.random.random()<CFG["channel_drop_prob"]: x[np.random.randint(0,3)]=0.0
        if np.random.random()<CFG["circ_shift_prob"]:
            x=np.roll(x,np.random.randint(-CFG["circ_shift_max"],CFG["circ_shift_max"]+1),1)
        if np.random.random()<CFG["amp_scale_prob"]:
            x=x*np.random.uniform(CFG["amp_scale_lo"],CFG["amp_scale_hi"])
        for _ in range(2):
            if np.random.random()<0.5:
                w=np.random.randint(1,CFG["time_mask_max"]+1); t0=np.random.randint(0,max(1,x.shape[1]-w)); x[:,t0:t0+w,:]=0.0
        for _ in range(2):
            if np.random.random()<0.5:
                w=np.random.randint(1,CFG["range_mask_max"]+1); r0=np.random.randint(0,max(1,x.shape[2]-w)); x[:,:,r0:r0+w]=0.0
        if np.random.random()<0.5: x=x+np.random.normal(0,CFG["noise_std"],x.shape).astype(np.float32)
        if np.random.random()<0.3:
            f=np.random.uniform(CFG["time_warp_lo"],CFG["time_warp_hi"]); nt=max(1,int(x.shape[1]*f))
            x=zoom(x,(1,nt/x.shape[1],1),order=1).astype(np.float32)
            x=self._pad_or_crop(x,mf,True)
        return x
    def __getitem__(self, idx):
        x=self.cache[self.keys[idx]].copy()
        x=(x-_MEANS)/_STDS
        x=self._pad_or_crop(x, self.max_frames, self.augment)
        if self.augment: x=self._apply_aug(x)
        t=torch.from_numpy(x)
        if self.img_size is not None:
            sz=(self.img_size, self.img_size) if isinstance(self.img_size, int) else self.img_size
            t=F.interpolate(t.unsqueeze(0),size=sz,mode="bilinear",align_corners=False).squeeze(0)
        if self.augment: t=_ERASER(t)
        soft=self.soft_labels[self.keys[idx]] if self.soft_labels else torch.zeros(CFG["n_classes"])
        return t, self.labels[idx], soft

class GeM(nn.Module):
    def __init__(self,p=3.0,eps=1e-6): super().__init__(); self.p=nn.Parameter(torch.ones(1)*p); self.eps=eps
    def forward(self,x): return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),1).pow(1.0/self.p)
class MultiSampleDropout(nn.Module):
    def __init__(self,inf,outf,n_samples=5,drop_rate=0.3):
        super().__init__(); self.dropouts=nn.ModuleList([nn.Dropout(drop_rate) for _ in range(n_samples)]); self.fc=nn.Linear(inf,outf)
    def forward(self,x): return torch.mean(torch.stack([self.fc(d(x)) for d in self.dropouts]),0) if self.training else self.fc(x)

class MultiScaleConvNeXtV2(nn.Module):
    def __init__(self, model_name, n_classes=126, pretrained=True,
                 drop_path_rate=0.2, proj_dim=512, drop_rate=0.3, ms_samples=5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True,
                                          out_indices=(0,1,2,3), drop_path_rate=drop_path_rate)
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

class MultiScaleCAFormer(nn.Module):
    def __init__(self, model_name, n_classes=126, pretrained=True,
                 drop_path_rate=0.15, proj_dim=384, drop_rate=0.3, ms_samples=5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True,
                                          out_indices=(0,1,2,3), drop_path_rate=drop_path_rate)
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

TEACHER_CONFIGS = [
    {"dir": VIT_V9_DIR, "cls": MultiScaleCAFormer, "model_name": "caformer_s18.sail_in22k_ft_in1k",
     "kwargs": {"drop_path_rate":0.0,"proj_dim":384,"drop_rate":0.3,"ms_samples":5}},
]

def load_born_again(model, cp, dev):
    ck=torch.load(cp,map_location=dev,weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    log.info("  Born-again ← %s (val %.2f%%)",cp.name,100*ck.get("val_acc",0))

def get_llrd_params(model, base_lr, decay=0.75, wd=0.05):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        if "backbone.stem" in name: lr=base_lr*decay**5
        elif "backbone.stages_0" in name: lr=base_lr*decay**4
        elif "backbone.stages_1" in name: lr=base_lr*decay**3
        elif "backbone.stages_2" in name: lr=base_lr*decay**2
        elif "backbone.stages_3" in name: lr=base_lr*decay**1
        else: lr=base_lr
        w=0.0 if p.ndim<2 else wd; key=(lr,w)
        if key not in groups: groups[key]={"params":[],"lr":lr,"weight_decay":w}
        groups[key]["params"].append(p)
    r=list(groups.values()); lrs=sorted(set(g["lr"] for g in r))
    log.info("LLRD: %d groups, LR [%.2e … %.2e]",len(r),min(lrs),max(lrs)); return r

def _load_teachers(configs, dev):
    ts=[]
    for cfg in configs:
        td=cfg["dir"]
        if not td.exists(): continue
        kw={"n_classes":CFG["n_classes"],"pretrained":False,**cfg["kwargs"]}
        for f in range(CFG["n_folds"]):
            cp=td/f"best_fold{f}.pt"
            if not cp.exists(): continue
            m=cfg["cls"](cfg["model_name"],**kw).to(dev)
            m.load_state_dict(torch.load(cp,map_location=dev,weights_only=True)["model_state_dict"])
            m.eval(); ts.append(m)
    return ts

def precompute_teacher_labels(keys, cache, configs, dev):
    log.info("Pre-computing teacher soft labels (at 224×224) …")
    ds=RadarDataset(keys,[0]*len(keys),cache,augment=False,
                    img_size=CFG["teacher_img_size"],max_frames=CFG["teacher_max_frames"])
    loader=DataLoader(ds,batch_size=96,shuffle=False,num_workers=CFG["num_workers"],pin_memory=True)
    ts=_load_teachers(configs,dev); log.info("  Loaded %d teachers",len(ts))
    ap=[]; nt=len(ts)
    with torch.no_grad():
        for x,_,_ in tqdm(loader,desc="  teacher fwd",leave=False):
            x=x.to(dev,non_blocking=True)
            ap.append((sum(F.softmax(m(x),1) for m in ts)/nt).cpu())
    sd={k:torch.cat(ap)[i] for i,k in enumerate(keys)}
    del ts; torch.cuda.empty_cache(); log.info("  Soft labels: %d",len(sd)); return sd

def rand_bbox(H,W,lam):
    cr=np.sqrt(1.0-lam); ch,cw=int(H*cr),int(W*cr)
    cy,cx=np.random.randint(H),np.random.randint(W)
    return max(0,cy-ch//2),min(H,cy+ch//2),max(0,cx-cw//2),min(W,cx+cw//2)
def mix_or_cut(x,y,soft):
    idx=torch.randperm(x.size(0),device=x.device)
    if np.random.random()<CFG["cutmix_prob"]:
        lam=float(np.random.beta(CFG["cutmix_alpha"],CFG["cutmix_alpha"]))
        t0,t1,r0,r1=rand_bbox(x.size(2),x.size(3),lam)
        xm=x.clone(); xm[:,:,t0:t1,r0:r1]=x[idx,:,t0:t1,r0:r1]
        lam=1.0-(t1-t0)*(r1-r0)/(x.size(2)*x.size(3))
    else:
        lam=float(np.random.beta(CFG["mixup_alpha"],CFG["mixup_alpha"]))
        xm=lam*x+(1-lam)*x[idx]
    return xm,y,y[idx],soft,soft[idx],lam

def kd_loss_fn(lo,st,tau):
    lp=F.log_softmax(lo/tau,1); s2=(st+1e-8).pow(1.0/tau); s2=s2/s2.sum(1,keepdim=True)
    return F.kl_div(lp,s2,reduction="batchmean")*(tau**2)

def train_epoch(model, loader, opt, sched, crit, dev, alpha, tau, use_mix=True):
    model.train(); tl,co,n=0.,0.,0
    for x,y,soft in tqdm(loader,desc="  train",leave=False):
        x,y,soft=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True),soft.to(dev,non_blocking=True)
        if use_mix: xm,ya,yb,sa,sb,lam=mix_or_cut(x,y,soft)
        else: xm,ya,yb,sa,sb,lam=x,y,y,soft,soft,1.0
        logits=model(xm)
        ce=lam*crit(logits,ya)+(1-lam)*crit(logits,yb)
        kd=lam*kd_loss_fn(logits,sa,tau)+(1-lam)*kd_loss_fn(logits,sb,tau)
        loss=alpha*ce+(1-alpha)*kd
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),CFG["grad_clip_norm"])
        opt.step()
        if sched is not None: sched.step()
        bs=x.size(0); tl+=loss.item()*bs
        co+=(lam*logits.argmax(1).eq(ya).float()+(1-lam)*logits.argmax(1).eq(yb).float()).sum().item()
        n+=bs
    return tl/n,co/n

@torch.no_grad()
def eval_epoch(model, loader, crit, dev):
    model.eval(); tl,co,n=0.,0,0
    for x,y,_ in tqdm(loader,desc="  val  ",leave=False):
        x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True)
        lo=model(x); tl+=crit(lo,y).item()*x.size(0); co+=lo.argmax(1).eq(y).sum().item(); n+=x.size(0)
    return tl/n,co/n

def _tta(x):
    return [x,x.flip(2),x.roll(-4,2),x.roll(4,2),x[:,[1,2,0],:,:],x[:,[2,0,1],:,:],
            x+.05*torch.randn_like(x),x+.08*torch.randn_like(x),x*.9,x*1.1]
@torch.no_grad()
def predict_tta(ms,loader,dev,tta=True):
    for m in ms: m.eval()
    ap=[]
    for x,_,_ in tqdm(loader,desc="  predict",leave=False):
        x=x.to(dev,non_blocking=True); bp=torch.zeros(x.size(0),CFG["n_classes"],device=dev)
        vs=_tta(x) if tta else [x]
        for v in vs:
            for m in ms: bp+=F.softmax(m(v),1)
        ap.append((bp/(len(vs)*len(ms))).cpu())
    return torch.cat(ap)
def _write_sub(keys,preds,path):
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["id","Pred"])
        for k,p in zip(keys,preds): w.writerow([k.split("_",1)[1],int(p)])

def main():
    out=Path(CFG["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.backends.cudnn.benchmark=True
    dev=torch.device("cuda:0"); log.info("GPU: %s",torch.cuda.get_device_name(0))
    log.info("Config: %s",json.dumps(CFG,indent=2))
    log.info("★★★ HIGH RESOLUTION: %s×%s ★★★", CFG["img_size"], CFG["img_size"])

    train_samples,idx_to_cls=collect_train_samples(CFG["data_dir"])
    eval_samples=collect_eval_samples(CFG["data_dir"])
    keys=[s[1] for s in train_samples]; labels=np.array([s[2] for s in train_samples])
    eval_keys=[s[1] for s in eval_samples]
    log.info("Train: %d | Eval: %d",len(keys),len(eval_samples))
    train_cache=preload(train_samples); eval_cache=preload(eval_samples,is_test=True)
    json.dump(CFG,open(out/"config.json","w"),indent=2)
    soft_labels=precompute_teacher_labels(keys,train_cache,TEACHER_CONFIGS,dev)
    alpha,tau=CFG["kd_alpha"],CFG["kd_tau"]
    log.info("KD α=%.2f τ=%.1f | SWA %d ep @ lr=%.1e | NO PL | NATIVE RES",alpha,tau,CFG["swa_epochs"],CFG["swa_lr"])

    skf=StratifiedKFold(n_splits=CFG["n_folds"],shuffle=True,random_state=CFG["seed"]); fold_accs=[]
    for fold,(tri,vai) in enumerate(skf.split(keys,labels)):
        log.info("="*64); log.info("FOLD %d / %d",fold+1,CFG["n_folds"]); log.info("="*64)
        trk=[keys[i] for i in tri]; trl=labels[tri].tolist()
        vak=[keys[i] for i in vai]; val_=labels[vai].tolist()
        log.info("  Train: %d | Val: %d",len(trk),len(vak))
        tds=RadarDataset(trk,trl,train_cache,augment=True,soft_labels=soft_labels,
                         img_size=CFG["img_size"],max_frames=CFG["max_frames"])
        vds=RadarDataset(vak,val_,train_cache,augment=False,
                         img_size=CFG["img_size"],max_frames=CFG["max_frames"])
        tl=DataLoader(tds,batch_size=CFG["batch_size"],shuffle=True,num_workers=CFG["num_workers"],
                      pin_memory=True,drop_last=True,persistent_workers=True)
        vl=DataLoader(vds,batch_size=CFG["batch_size"]*2,shuffle=False,num_workers=CFG["num_workers"],
                      pin_memory=True,persistent_workers=True)

        model=MultiScaleConvNeXtV2(CFG["model_name"],CFG["n_classes"],pretrained=True,
                                   drop_path_rate=CFG["drop_path_rate"],proj_dim=CFG["proj_dim"],
                                   drop_rate=CFG["drop_rate"],ms_samples=CFG["ms_dropout_samples"]).to(dev)
        log.info("  Using fresh ImageNet pretrained ConvNeXtV2-Base (no born-again)")
        crit=nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])
        best_acc=0.0; ckpt=out/f"best_fold{fold}.pt"

        # Stage 1: head warmup
        log.info("── Stage 1: head warmup (%d ep) ──",CFG["stage1_epochs"])
        for p in model.backbone.parameters(): p.requires_grad=False
        s1p=[p for p in model.parameters() if p.requires_grad]
        opt=AdamW(s1p,lr=CFG["stage1_lr"],weight_decay=CFG["weight_decay"])
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG["stage1_lr"],
            total_steps=len(tl)*CFG["stage1_epochs"],pct_start=0.3,anneal_strategy="cos",div_factor=10,final_div_factor=100)
        for ep in range(1,CFG["stage1_epochs"]+1):
            t0=time.time()
            trl_,tra_=train_epoch(model,tl,opt,sched,crit,dev,alpha,tau,use_mix=False)
            _,va_=eval_epoch(model,vl,crit,dev)
            log.info("  S1 Ep %d/%d  loss=%.4f acc=%.1f%%  val=%.1f%%  %.0fs",ep,CFG["stage1_epochs"],trl_,100*tra_,100*va_,time.time()-t0)
            if va_>best_acc:
                best_acc=va_; torch.save({"fold":fold,"epoch":ep,"stage":1,"model_state_dict":model.state_dict(),"val_acc":va_},ckpt)
                log.info("  ★ best %.2f%%",100*va_)
        del opt,sched

        # Stage 2: full fine-tune + LLRD
        log.info("── Stage 2: full fine-tune (%d ep, patience %d) ──",CFG["stage2_epochs"],CFG["early_stopping_patience"])
        for p in model.parameters(): p.requires_grad=True
        pg=get_llrd_params(model,CFG["stage2_lr"],decay=CFG["llrd_decay"],wd=CFG["weight_decay"])
        opt=AdamW(pg,lr=CFG["stage2_lr"],weight_decay=CFG["weight_decay"])
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[g["lr"] for g in pg],
            total_steps=len(tl)*CFG["stage2_epochs"],pct_start=CFG["warmup_pct"],anneal_strategy="cos",div_factor=25,final_div_factor=1000)
        pat=0
        for ep in range(1,CFG["stage2_epochs"]+1):
            t0=time.time()
            trl_,tra_=train_epoch(model,tl,opt,sched,crit,dev,alpha,tau,use_mix=True)
            _,va_=eval_epoch(model,vl,crit,dev)
            log.info("  S2 Ep %3d/%d  loss=%.4f acc=%.1f%%  val=%.1f%%  lr=%.2e  %4.0fs",
                     ep,CFG["stage2_epochs"],trl_,100*tra_,100*va_,opt.param_groups[-1]["lr"],time.time()-t0)
            if va_>best_acc:
                best_acc=va_; pat=0
                torch.save({"fold":fold,"epoch":ep,"stage":2,"model_state_dict":model.state_dict(),"val_acc":va_},ckpt)
                log.info("  ★ best %.2f%%",100*va_)
            else:
                pat+=1
                if pat>=CFG["early_stopping_patience"]: log.info("  Early stop at ep %d",ep); break
        del opt,sched

        # Stage 3: SWA from best checkpoint
        log.info("── Stage 3: SWA (%d ep, lr=%.1e) ──",CFG["swa_epochs"],CFG["swa_lr"])
        model.load_state_dict(torch.load(ckpt,map_location=dev,weights_only=True)["model_state_dict"])
        swa_model=AveragedModel(model)
        swa_opt=AdamW(model.parameters(),lr=CFG["swa_lr"],weight_decay=CFG["weight_decay"])
        for swa_ep in range(1,CFG["swa_epochs"]+1):
            t0=time.time()
            trl_,_=train_epoch(model,tl,swa_opt,None,crit,dev,alpha,tau,use_mix=True)
            swa_model.update_parameters(model)
            _,sva=eval_epoch(swa_model,vl,crit,dev)
            log.info("  SWA Ep %2d/%d  loss=%.4f  swa_val=%.1f%%  %.0fs",
                     swa_ep,CFG["swa_epochs"],trl_,100*sva,time.time()-t0)
        log.info("  Updating BN stats for SWA model …")
        update_bn(tl, swa_model, device=dev)
        _,swa_acc=eval_epoch(swa_model,vl,crit,dev)
        log.info("  SWA final: %.2f%% (regular best: %.2f%%)",100*swa_acc,100*best_acc)
        if swa_acc>best_acc:
            best_acc=swa_acc
            torch.save({"fold":fold,"epoch":-1,"stage":3,"model_state_dict":swa_model.module.state_dict(),"val_acc":swa_acc},ckpt)
            log.info("  ★ SWA beats regular! %.2f%%",100*swa_acc)
        fold_accs.append(best_acc); log.info("  Fold %d best: %.2f%%",fold+1,100*best_acc)
        del model,swa_model,swa_opt,tl,vl; torch.cuda.empty_cache()

    ma,sa=np.mean(fold_accs),np.std(fold_accs)
    log.info("="*64); log.info("5-FOLD CV"); log.info("="*64)
    for i,a in enumerate(fold_accs): log.info("  Fold %d : %.2f%%",i+1,100*a)
    log.info("  Mean: %.2f%% ± %.2f%%",100*ma,100*sa)
    json.dump({"fold_accuracies":[float(a) for a in fold_accs],"mean":float(ma),"std":float(sa)},open(out/"cv_results.json","w"),indent=2)

    eds=RadarDataset(eval_keys,[0]*len(eval_keys),eval_cache,augment=False,
                     img_size=CFG["img_size"],max_frames=CFG["max_frames"])
    el=DataLoader(eds,batch_size=CFG["batch_size"],shuffle=False,num_workers=CFG["num_workers"],pin_memory=True)
    fms=[]
    for f in range(CFG["n_folds"]):
        m=MultiScaleConvNeXtV2(CFG["model_name"],CFG["n_classes"],pretrained=False,drop_path_rate=0.0,
                               proj_dim=CFG["proj_dim"],drop_rate=CFG["drop_rate"],ms_samples=CFG["ms_dropout_samples"]).to(dev)
        m.load_state_dict(torch.load(out/f"best_fold{f}.pt",map_location=dev,weights_only=True)["model_state_dict"]); fms.append(m)
    pn=predict_tta(fms,el,dev,tta=False); _write_sub(eval_keys,pn.argmax(1).numpy(),out/"submission_no_tta.csv")
    pt=predict_tta(fms,el,dev,tta=True); _write_sub(eval_keys,pt.argmax(1).numpy(),out/"submission_tta.csv")
    ch=(pt.argmax(1).numpy()!=pn.argmax(1).numpy()).sum()
    log.info("TTA changed %d / %d (%.1f%%)",ch,len(eval_keys),100*ch/len(eval_keys))
    torch.save({"keys":eval_keys,"probs":pt},out/"eval_probs_tta.pt"); log.info("Done.")

if __name__=="__main__": main()
