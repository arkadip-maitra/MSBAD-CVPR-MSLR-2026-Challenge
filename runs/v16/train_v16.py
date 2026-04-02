#!/usr/bin/env python3
"""
v16: ConvNeXt-Small (50M params) + SupCon + SAM — NEW BIGGER BACKBONE
NO born-again (new architecture), cross-arch KD from v11 + vit_v9
ConvNeXt-Small has 2x params vs Tiny → more capacity for fine-grained gestures.
Different from ConvNeXtV2 (no GRN) → adds ensemble diversity.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomErasing
from tqdm import tqdm

OUT_DIR = Path("/data/slr/checkpoints_v16")
V11_DIR = Path("/data/slr/checkpoints_v11")
VIT_V9_DIR = Path("/data/slr/checkpoints_vit_v9")
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(OUT_DIR/"training.log"), logging.StreamHandler()])
log = logging.getLogger(__name__)

CFG = {
    "data_dir": "/data/slr/track2_", "output_dir": str(OUT_DIR),
    "max_frames": 48, "n_classes": 126,
    "channel_means": [-64.3217, -62.9119, -63.9254],
    "channel_stds": [12.5721, 11.6620, 11.7259],
    "n_folds": 5, "batch_size": 48, "num_workers": 4, "img_size": 224,
    "stage1_epochs": 8, "stage1_lr": 2e-4,
    "stage2_epochs": 120, "stage2_lr": 2e-5,
    "weight_decay": 0.05, "warmup_pct": 0.08,
    "label_smoothing": 0.1, "early_stopping_patience": 40,
    "grad_clip_norm": 1.0, "llrd_decay": 0.80,
    "sam_rho": 0.05,
    "ce_weight": 0.4, "supcon_weight": 0.3, "kd_weight": 0.3,
    "kd_tau": 4.0, "supcon_temp": 0.1, "proj_out_dim": 128,
    "time_mask_max": 8, "range_mask_max": 30, "noise_std": 0.1,
    "time_warp_lo": 0.8, "time_warp_hi": 1.2,
    "time_reverse_prob": 0.5, "channel_shuffle_prob": 0.2, "channel_drop_prob": 0.1,
    "random_erase_prob": 0.20, "circ_shift_prob": 0.3, "circ_shift_max": 8,
    "amp_scale_prob": 0.5, "amp_scale_lo": 0.8, "amp_scale_hi": 1.2,
    "model_name": "convnext_small.fb_in22k_ft_in1k",
    "drop_path_rate": 0.3,
    "proj_dim": 512, "drop_rate": 0.4,
    "ms_dropout_samples": 5, "seed": 42,
}

# ── data ──
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
def preload(samples,is_test=False):
    c={}
    for item in tqdm(samples,desc="Pre-loading",leave=False): c[item[1]]=_load_one(item[0],item[1])
    return c

_MEANS=np.array(CFG["channel_means"],dtype=np.float32).reshape(3,1,1)
_STDS=np.array(CFG["channel_stds"],dtype=np.float32).reshape(3,1,1)
_ERASER=RandomErasing(p=CFG["random_erase_prob"],scale=(0.02,0.2),ratio=(0.3,3.3),value=0)

class RadarDataset(Dataset):
    def __init__(s,keys,labels,cache,augment=False,soft_labels=None,img_size=224):
        s.keys,s.labels,s.cache,s.augment,s.soft_labels,s.img_size=keys,labels,cache,augment,soft_labels,img_size
    def __len__(s): return len(s.keys)
    @staticmethod
    def _pad_or_crop(x,mt,rc=False):
        t=x.shape[1]
        if t>mt: st=np.random.randint(0,t-mt+1) if rc else (t-mt)//2; x=x[:,st:st+mt,:]
        elif t<mt: x=np.tile(x,(1,mt//t+1,1))[:,:mt,:]
        return x
    def _apply_aug(s,x):
        if np.random.random()<CFG["time_reverse_prob"]: x=x[:,::-1,:].copy()
        if np.random.random()<CFG["channel_shuffle_prob"]: x=x[np.random.permutation(3)]
        if np.random.random()<CFG["channel_drop_prob"]: x[np.random.randint(0,3)]=0.0
        if np.random.random()<CFG["circ_shift_prob"]: x=np.roll(x,np.random.randint(-CFG["circ_shift_max"],CFG["circ_shift_max"]+1),1)
        if np.random.random()<CFG["amp_scale_prob"]: x=x*np.random.uniform(CFG["amp_scale_lo"],CFG["amp_scale_hi"])
        for _ in range(2):
            if np.random.random()<0.5: w=np.random.randint(1,CFG["time_mask_max"]+1); t0=np.random.randint(0,max(1,x.shape[1]-w)); x[:,t0:t0+w,:]=0.0
        for _ in range(2):
            if np.random.random()<0.5: w=np.random.randint(1,CFG["range_mask_max"]+1); r0=np.random.randint(0,max(1,x.shape[2]-w)); x[:,:,r0:r0+w]=0.0
        if np.random.random()<0.5: x=x+np.random.normal(0,CFG["noise_std"],x.shape).astype(np.float32)
        if np.random.random()<0.3:
            f=np.random.uniform(CFG["time_warp_lo"],CFG["time_warp_hi"]); nt=max(1,int(x.shape[1]*f))
            x=zoom(x,(1,nt/x.shape[1],1),order=1).astype(np.float32); x=s._pad_or_crop(x,CFG["max_frames"],True)
        return x
    def __getitem__(s,i):
        x=s.cache[s.keys[i]].copy(); x=(x-_MEANS)/_STDS; x=s._pad_or_crop(x,CFG["max_frames"],s.augment)
        if s.augment: x=s._apply_aug(x)
        t=torch.from_numpy(x); sz=(s.img_size,s.img_size) if isinstance(s.img_size,int) else s.img_size
        t=F.interpolate(t.unsqueeze(0),size=sz,mode="bilinear",align_corners=False).squeeze(0)
        if s.augment: t=_ERASER(t)
        soft=s.soft_labels[s.keys[i]] if s.soft_labels else torch.zeros(CFG["n_classes"])
        return t,s.labels[i],soft

# ── models ──
class GeM(nn.Module):
    def __init__(s,p=3.0,eps=1e-6): super().__init__(); s.p=nn.Parameter(torch.ones(1)*p); s.eps=eps
    def forward(s,x): return F.adaptive_avg_pool2d(x.clamp(min=s.eps).pow(s.p),1).pow(1.0/s.p)
class MultiSampleDropout(nn.Module):
    def __init__(s,inf,outf,ns=5,dr=0.3): super().__init__(); s.dropouts=nn.ModuleList([nn.Dropout(dr) for _ in range(ns)]); s.fc=nn.Linear(inf,outf)
    def forward(s,x): return torch.mean(torch.stack([s.fc(d(x)) for d in s.dropouts]),0) if s.training else s.fc(x)
def _build_ms(bb,pd,nc,dr,ms,pod):
    sd=bb.feature_info.channels(); g=nn.ModuleList([GeM(3.0) for _ in sd])
    p=nn.ModuleList([nn.Sequential(nn.Linear(d,pd),nn.LayerNorm(pd),nn.GELU()) for d in sd])
    a=nn.Sequential(nn.Linear(pd,1)); h=MultiSampleDropout(pd,nc,ms,dr)
    ph=nn.Sequential(nn.Linear(pd,pd),nn.GELU(),nn.Linear(pd,pod)) if pod>0 else None
    return g,p,a,h,ph
def _fwd_ms(bb,g,p,a,h,ph,x,rp=False):
    fs=bb(x); ps=[pj(gm(f).flatten(1)) for gm,pj,f in zip(g,p,fs)]
    st=torch.stack(ps,1); w=F.softmax(a(st),1); ag=(st*w).sum(1); lo=h(ag)
    if rp and ph is not None: return lo,F.normalize(ph(ag),dim=1)
    return lo

class MultiScaleConvNeXt(nn.Module):
    """Works for both convnext_small and convnextv2_tiny (same stage dims)."""
    def __init__(s,mn,nc=126,pt=True,dpr=0.3,pd=512,dr=0.4,ms=5,pod=0):
        super().__init__(); s.backbone=timm.create_model(mn,pretrained=pt,features_only=True,out_indices=(0,1,2,3),drop_path_rate=dpr)
        s.stage_gems,s.stage_projs,s.scale_attn,s.head,s.proj_head=_build_ms(s.backbone,pd,nc,dr,ms,pod)
    def forward(s,x,return_proj=False): return _fwd_ms(s.backbone,s.stage_gems,s.stage_projs,s.scale_attn,s.head,s.proj_head,x,return_proj)

class MultiScaleConvNeXtV2(nn.Module):
    def __init__(s,mn,nc=126,pt=True,dpr=0.0,pd=512,dr=0.3,ms=5,pod=0):
        super().__init__(); s.backbone=timm.create_model(mn,pretrained=pt,features_only=True,out_indices=(0,1,2,3),drop_path_rate=dpr)
        s.stage_gems,s.stage_projs,s.scale_attn,s.head,s.proj_head=_build_ms(s.backbone,pd,nc,dr,ms,pod)
    def forward(s,x,return_proj=False): return _fwd_ms(s.backbone,s.stage_gems,s.stage_projs,s.scale_attn,s.head,s.proj_head,x,return_proj)

class MultiScaleCAFormer(nn.Module):
    def __init__(s,mn,nc=126,pt=True,dpr=0.0,pd=384,dr=0.3,ms=5,pod=0):
        super().__init__(); s.backbone=timm.create_model(mn,pretrained=pt,features_only=True,out_indices=(0,1,2,3),drop_path_rate=dpr)
        s.stage_gems,s.stage_projs,s.scale_attn,s.head,s.proj_head=_build_ms(s.backbone,pd,nc,dr,ms,pod)
    def forward(s,x,return_proj=False): return _fwd_ms(s.backbone,s.stage_gems,s.stage_projs,s.scale_attn,s.head,s.proj_head,x,return_proj)

# ── SAM + SupCon ──
class SAM:
    def __init__(s,bo,rho=0.05): s.base_optimizer=bo; s.rho=rho; s._eps={}
    @property
    def param_groups(s): return s.base_optimizer.param_groups
    @torch.no_grad()
    def first_step(s):
        gn=s._grad_norm(); sc=s.rho/(gn+1e-12)
        for g in s.base_optimizer.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                e=p.grad*sc; p.add_(e); s._eps[p]=e
    @torch.no_grad()
    def second_step(s):
        for g in s.base_optimizer.param_groups:
            for p in g["params"]:
                if p in s._eps: p.sub_(s._eps[p])
        s.base_optimizer.step(); s._eps.clear()
    def zero_grad(s,set_to_none=False): s.base_optimizer.zero_grad(set_to_none=set_to_none)
    def _grad_norm(s):
        ns=[p.grad.detach().norm(2) for g in s.base_optimizer.param_groups for p in g["params"] if p.grad is not None]
        return torch.norm(torch.stack(ns),2) if ns else torch.tensor(0.0)

class SupConLoss(nn.Module):
    def __init__(s,temperature=0.1): super().__init__(); s.temperature=temperature
    def forward(s,features,labels):
        dev=features.device; B=features.shape[0]
        if B<=1: return torch.tensor(0.0,device=dev,requires_grad=True)
        sim=torch.matmul(features,features.T)/s.temperature
        sm=torch.eye(B,dtype=torch.bool,device=dev)
        pm=(labels.unsqueeze(0)==labels.unsqueeze(1)).float().masked_fill(sm,0)
        sim=sim-sim.max(1,keepdim=True).values.detach()
        es=torch.exp(sim).masked_fill(sm,0)
        lp=sim-torch.log(es.sum(1,keepdim=True)+1e-8)
        pc=pm.sum(1); v=pc>0
        if v.sum()==0: return torch.tensor(0.0,device=dev,requires_grad=True)
        return -(pm*lp).sum(1)[v].div(pc[v]+1e-8).mean()

# Teachers: v11 (ConvNeXtV2) + vit_v9 (CAFormer) for cross-arch KD
TEACHER_CONFIGS = [
    {"dir": V11_DIR, "cls": MultiScaleConvNeXtV2, "model_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
     "kwargs": {"dpr":0.0,"pd":512,"dr":0.3,"ms":5}},
    {"dir": VIT_V9_DIR, "cls": MultiScaleCAFormer, "model_name": "caformer_s18.sail_in22k_ft_in1k",
     "kwargs": {"dpr":0.0,"pd":384,"dr":0.3,"ms":5}},
]

def get_llrd_params(model,base_lr,decay=0.80,wd=0.05):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        if "backbone.stem" in name:       lr=base_lr*decay**5
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

def _load_teachers(configs,dev):
    ts=[]
    for cfg in configs:
        td=cfg["dir"]
        if not td.exists(): continue
        kw={"nc":CFG["n_classes"],"pt":False,**cfg["kwargs"]}
        for f in range(CFG["n_folds"]):
            cp=td/f"best_fold{f}.pt"
            if not cp.exists(): continue
            m=cfg["cls"](cfg["model_name"],**kw).to(dev)
            m.load_state_dict(torch.load(cp,map_location=dev,weights_only=True)["model_state_dict"])
            m.eval(); ts.append(m)
    return ts
def _teacher_infer(ts,loader,dev):
    ap=[]; nt=len(ts)
    with torch.no_grad():
        for x,_,_ in tqdm(loader,desc="  teacher fwd",leave=False):
            x=x.to(dev,non_blocking=True); ap.append((sum(F.softmax(m(x),1) for m in ts)/nt).cpu())
    return torch.cat(ap)
def precompute_teacher_labels(keys,cache,cfgs,dev):
    log.info("Pre-computing teacher soft labels …")
    ds=RadarDataset(keys,[0]*len(keys),cache,augment=False,img_size=CFG["img_size"])
    loader=DataLoader(ds,batch_size=96,shuffle=False,num_workers=CFG["num_workers"],pin_memory=True)
    ts=_load_teachers(cfgs,dev); log.info("  Loaded %d teachers",len(ts))
    sa=_teacher_infer(ts,loader,dev); sd={k:sa[i] for i,k in enumerate(keys)}
    del ts; torch.cuda.empty_cache(); log.info("  Soft labels: %d",len(sd)); return sd

def kd_loss_fn(lo,st,tau):
    lp=F.log_softmax(lo/tau,1); s2=(st+1e-8).pow(1.0/tau); s2=s2/s2.sum(1,keepdim=True)
    return F.kl_div(lp,s2,reduction="batchmean")*(tau**2)

def _compute_loss(model,x,y,soft,crit,sc_crit,cfg,tau):
    lo,pr=model(x,return_proj=True); ce=crit(lo,y); sc=sc_crit(pr,y); kd=kd_loss_fn(lo,soft,tau)
    return lo, cfg["ce_weight"]*ce+cfg["supcon_weight"]*sc+cfg["kd_weight"]*kd, ce.item(), sc.item()

def train_epoch(model,loader,opt,sched,crit,sc,dev,cfg,tau,is_sam=False):
    model.train(); tl,tc,tsc,co,n=0.,0.,0.,0.,0
    for x,y,soft in tqdm(loader,desc="  train",leave=False):
        x,y,soft=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True),soft.to(dev,non_blocking=True)
        lo,loss,cev,scv=_compute_loss(model,x,y,soft,crit,sc,cfg,tau); loss.backward()
        if is_sam:
            opt.first_step(); opt.zero_grad(set_to_none=True)
            _,l2,_,_=_compute_loss(model,x,y,soft,crit,sc,cfg,tau); l2.backward()
            nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip_norm"])
            opt.second_step(); opt.zero_grad(set_to_none=True)
        else:
            nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip_norm"]); opt.step(); opt.zero_grad(set_to_none=True)
        sched.step(); bs=x.size(0); tl+=loss.item()*bs; tc+=cev*bs; tsc+=scv*bs
        co+=lo.argmax(1).eq(y).sum().item(); n+=bs
    return tl/n,co/n,tc/n,tsc/n

@torch.no_grad()
def eval_epoch(model,loader,crit,dev):
    model.eval(); tl,co,n=0.,0,0
    for x,y,_ in tqdm(loader,desc="  val  ",leave=False):
        x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True)
        lo=model(x); tl+=crit(lo,y).item()*x.size(0); co+=lo.argmax(1).eq(y).sum().item(); n+=x.size(0)
    return tl/n,co/n

def _tta(x): return [x,x.flip(2),x.roll(-8,2),x.roll(8,2),x[:,[1,2,0],:,:],x[:,[2,0,1],:,:],x+.05*torch.randn_like(x),x+.08*torch.randn_like(x),x*.9,x*1.1]
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
    train_samples,idx_to_cls=collect_train_samples(CFG["data_dir"])
    eval_samples=collect_eval_samples(CFG["data_dir"])
    keys=[s[1] for s in train_samples]; labels=np.array([s[2] for s in train_samples])
    eval_keys=[s[1] for s in eval_samples]
    log.info("Train: %d | Eval: %d | Classes: %d",len(keys),len(eval_samples),len(idx_to_cls))
    train_cache=preload(train_samples); eval_cache=preload(eval_samples,is_test=True)
    json.dump(CFG,open(out/"config.json","w"),indent=2)
    soft_labels=precompute_teacher_labels(keys,train_cache,TEACHER_CONFIGS,dev)

    probe=MultiScaleConvNeXt(CFG["model_name"],nc=CFG["n_classes"],pt=False,
                             dpr=CFG["drop_path_rate"],pd=CFG["proj_dim"],dr=CFG["drop_rate"],
                             ms=CFG["ms_dropout_samples"],pod=CFG["proj_out_dim"])
    log.info("ConvNeXt-Small: %.2fM params",sum(p.numel() for p in probe.parameters())/1e6); del probe
    log.info("Loss: %.1f*CE + %.1f*SupCon + %.1f*KD | SAM | NO PL | NO born-again",
             CFG["ce_weight"],CFG["supcon_weight"],CFG["kd_weight"])

    skf=StratifiedKFold(n_splits=CFG["n_folds"],shuffle=True,random_state=CFG["seed"]); fold_accs=[]
    for fold,(tri,vai) in enumerate(skf.split(keys,labels)):
        log.info("="*64); log.info("FOLD %d / %d",fold+1,CFG["n_folds"]); log.info("="*64)
        trk=[keys[i] for i in tri]; trl=labels[tri].tolist()
        vak=[keys[i] for i in vai]; val_=labels[vai].tolist()
        log.info("  Train: %d | Val: %d",len(trk),len(vak))
        tds=RadarDataset(trk,trl,train_cache,augment=True,soft_labels=soft_labels,img_size=CFG["img_size"])
        vds=RadarDataset(vak,val_,train_cache,augment=False,img_size=CFG["img_size"])
        tl=DataLoader(tds,batch_size=CFG["batch_size"],shuffle=True,num_workers=CFG["num_workers"],pin_memory=True,drop_last=True,persistent_workers=True)
        vl=DataLoader(vds,batch_size=CFG["batch_size"]*2,shuffle=False,num_workers=CFG["num_workers"],pin_memory=True,persistent_workers=True)

        model=MultiScaleConvNeXt(CFG["model_name"],nc=CFG["n_classes"],pt=True,
                                 dpr=CFG["drop_path_rate"],pd=CFG["proj_dim"],
                                 dr=CFG["drop_rate"],ms=CFG["ms_dropout_samples"],
                                 pod=CFG["proj_out_dim"]).to(dev)
        log.info("  Using ImageNet pretrained weights (no born-again)")
        crit=nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"]); sc=SupConLoss(CFG["supcon_temp"])
        best_acc=0.0; ckpt=out/f"best_fold{fold}.pt"

        # Stage 1: head+proj warmup (longer for from-scratch)
        log.info("── Stage 1: head+proj warmup (%d ep) ──",CFG["stage1_epochs"])
        for p in model.backbone.parameters(): p.requires_grad=False
        s1p=[p for p in model.parameters() if p.requires_grad]
        opt=AdamW(s1p,lr=CFG["stage1_lr"],weight_decay=CFG["weight_decay"])
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=CFG["stage1_lr"],total_steps=len(tl)*CFG["stage1_epochs"],pct_start=0.3,anneal_strategy="cos",div_factor=10.0,final_div_factor=100.0)
        for ep in range(1,CFG["stage1_epochs"]+1):
            t0=time.time()
            trl_,tra_,cev,scv=train_epoch(model,tl,opt,sched,crit,sc,dev,CFG,CFG["kd_tau"],False)
            _,va_=eval_epoch(model,vl,crit,dev)
            log.info("  S1 Ep %d/%d  loss=%.4f acc=%.1f%% ce=%.3f sc=%.3f  val=%.1f%%  %.0fs",ep,CFG["stage1_epochs"],trl_,100*tra_,cev,scv,100*va_,time.time()-t0)
            if va_>best_acc:
                best_acc=va_; torch.save({"fold":fold,"epoch":ep,"stage":1,"model_state_dict":model.state_dict(),"val_acc":va_},ckpt)
                log.info("  ★ best %.2f%%",100*va_)
        del opt,sched

        # Stage 2: full fine-tune + SAM + LLRD
        log.info("── Stage 2: SAM+LLRD+SupCon (%d ep, patience %d) ──",CFG["stage2_epochs"],CFG["early_stopping_patience"])
        for p in model.parameters(): p.requires_grad=True
        pg=get_llrd_params(model,CFG["stage2_lr"],decay=CFG["llrd_decay"],wd=CFG["weight_decay"])
        bo=AdamW(pg,lr=CFG["stage2_lr"],weight_decay=CFG["weight_decay"]); so=SAM(bo,rho=CFG["sam_rho"])
        sched=torch.optim.lr_scheduler.OneCycleLR(bo,max_lr=[g["lr"] for g in pg],total_steps=len(tl)*CFG["stage2_epochs"],pct_start=CFG["warmup_pct"],anneal_strategy="cos",div_factor=25.0,final_div_factor=1000.0)
        pat=0
        for ep in range(1,CFG["stage2_epochs"]+1):
            t0=time.time()
            trl_,tra_,cev,scv=train_epoch(model,tl,so,sched,crit,sc,dev,CFG,CFG["kd_tau"],True)
            _,va_=eval_epoch(model,vl,crit,dev)
            log.info("  S2 Ep %3d/%d  loss=%.4f acc=%.1f%% ce=%.3f sc=%.3f  val=%.1f%%  lr=%.2e  %4.0fs",ep,CFG["stage2_epochs"],trl_,100*tra_,cev,scv,100*va_,bo.param_groups[-1]["lr"],time.time()-t0)
            if va_>best_acc:
                best_acc=va_; pat=0
                torch.save({"fold":fold,"epoch":ep,"stage":2,"model_state_dict":model.state_dict(),"val_acc":va_},ckpt)
                log.info("  ★ best %.2f%%",100*va_)
            else:
                pat+=1
                if pat>=CFG["early_stopping_patience"]: log.info("  Early stop at ep %d",ep); break
        fold_accs.append(best_acc); log.info("  Fold %d best: %.2f%%",fold+1,100*best_acc)
        del model,bo,so,sched,tl,vl; torch.cuda.empty_cache()

    ma,sa=np.mean(fold_accs),np.std(fold_accs)
    log.info("="*64); log.info("5-FOLD CV"); log.info("="*64)
    for i,a in enumerate(fold_accs): log.info("  Fold %d : %.2f%%",i+1,100*a)
    log.info("  Mean: %.2f%% ± %.2f%%",100*ma,100*sa)
    json.dump({"fold_accuracies":[float(a) for a in fold_accs],"mean":float(ma),"std":float(sa)},open(out/"cv_results.json","w"),indent=2)

    eds=RadarDataset(eval_keys,[0]*len(eval_keys),eval_cache,augment=False,img_size=CFG["img_size"])
    el=DataLoader(eds,batch_size=CFG["batch_size"],shuffle=False,num_workers=CFG["num_workers"],pin_memory=True)
    fms=[]
    for f in range(CFG["n_folds"]):
        m=MultiScaleConvNeXt(CFG["model_name"],nc=CFG["n_classes"],pt=False,dpr=0.0,pd=CFG["proj_dim"],dr=CFG["drop_rate"],ms=CFG["ms_dropout_samples"],pod=CFG["proj_out_dim"]).to(dev)
        m.load_state_dict(torch.load(out/f"best_fold{f}.pt",map_location=dev,weights_only=True)["model_state_dict"]); fms.append(m)
    pn=predict_tta(fms,el,dev,tta=False); _write_sub(eval_keys,pn.argmax(1).numpy(),out/"submission_no_tta.csv")
    pt=predict_tta(fms,el,dev,tta=True); _write_sub(eval_keys,pt.argmax(1).numpy(),out/"submission_tta.csv")
    ch=(pt.argmax(1).numpy()!=pn.argmax(1).numpy()).sum()
    log.info("TTA changed %d / %d (%.1f%%)",ch,len(eval_keys),100*ch/len(eval_keys))
    torch.save({"keys":eval_keys,"probs":pt},out/"eval_probs_tta.pt"); log.info("Done.")

if __name__=="__main__": main()
