# MSBAD at SignEval 2026: Multi-Scale Born-Again Distillation Ensemble for Radar-Based Sign Language Recognition

**CVPR MSLR 2026 Challenge -- Track 2 (Radar-Only)**

Final test accuracy: **89.91%**

## Overview

This repository contains all code and training logs for our solution to the SignEval 2026 Challenge Track 2: Italian Sign Language (LIS) recognition from 60 GHz radar Range-Time Maps (RTMs). The task is to classify 126 sign classes from three-channel radar data captured by spatially separated receivers.

Our approach combines:
- **Multi-scale feature extraction** using ConvNeXtV2-Tiny with GeM pooling and learned scale attention
- **Born-Again distillation** for iterative self-improvement over 3-4 generations
- **Multi-resolution training** (224px to 512px) for ensemble diversity
- **Multi-architecture ensemble** (ConvNeXtV2, EfficientNetV2, CAFormer)
- **Consensus filtering** on top of geometric-mean probability fusion

## Checkpoint Folder
[Drive Link](https://drive.google.com/drive/folders/1mE_hN_jbSE0VlUWizDjD_RlbIa8BkSDt?usp=sharing)

### Highlighted Runs

| Run | Type | Architecture | Resolution | Val Acc | Notes |
|-----|------|-------------|-----------|---------|-------|
| **v67** | Single-fold | ConvNeXtV2-Tiny | 224 | **90.98%** | Best single-model validation accuracy |
| **v113** | 5-fold | ConvNeXtV2-Tiny + Focal Loss | 288 | **88.14% avg** | Best 5-fold cross-validation average |

### Final Ensemble (89.91% test)

The best submission uses a 6-model ensemble with geometric-mean fusion and consensus filtering

## Repository Structure

```
.
├── README.md
├── best_submission.csv            # Best submission file (89.91% private)
├── consensus_3plus.csv            # 6-model geo + consensus (89.906% private)
├── submit1_best5_plus_v38_geo.csv # 6-model geo mean (89.906% private)
├── submit1_best8_geo.csv          # 8-model geo mean (89.845% private)
├── submit2_best5_plus_v38_v33_geo.csv # 7-model geo mean (89.845% private)
│
├── # --- Best Ensemble Code (top level) ---
├── train_v11.py                   # ConvNeXtV2-T @ 224
├── train_v18.py                   # ConvNeXtV2-T @ 384
├── train_v24.py                   # EfficientNetV2-S @ 384
├── train_v25.py                   # ConvNeXtV2-T @ 448
├── train_v38.py                   # ConvNeXtV2-T @ 512
├── train_vit_v9.py                # CAFormer-S18 @ 224
├── ensemble.py                    # Ensemble + consensus generation
│
├── # --- Ensemble Utilities ---
├── ensemble_final.py
├── ensemble_multiresolution.py
├── ensemble_weighted.py
├── mega_ensemble.py
├── mega_ensemble_v2.py
├── compute_oof.py
├── oof_stacker.py
├── multires_inference.py
│
└── runs/                          # All 131 experiment runs
    ├── v2/ ... v97/               # Single-fold ConvNeXt runs
    ├── v100_5fold/ ... v117_5fold/# 5-fold cross-validation runs
    ├── vit/ ... vit_v15/          # CAFormer/ViT runs
    └── 1d/, 1d_v2/               # 1D baseline runs
```

Each run folder contains:
- `train_*.py` -- Training script with all hyperparameters
- `training.log` or `training_fold*.log` -- Full training log with per-epoch metrics
- `config.json` -- Model configuration (where available)

## All Runs -- Results Summary

### 5-Fold Cross-Validation Runs

| Run | Architecture | Resolution | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Avg | Key Change |
|-----|-------------|-----------|--------|--------|--------|--------|--------|-----|------------|
| **v117** | ConvNeXtV2-T + Focal | 384 | **89.22** | 87.18 | 87.42 | 87.79 | 87.79 | **87.88** | Higher resolution, born-again from v113 |
| v116 | ConvNeXtV2-T + Focal | 288 | 88.54 | 87.45 | -- | -- | -- | -- | Extended 600ep from v113 |
| v106 | ConvNeXtV2-T | 224 | 87.15 | 86.64 | 87.79 | 87.25 | 86.77 | 87.12 | Weak aug, no early stop, 400ep |
| v107 | ConvNeXtV2-T | 224 | 87.39 | -- | -- | -- | -- | -- | Higher LR from v106 |
| v111 | ConvNeXtV2-T | 224 | 87.28 | -- | -- | -- | -- | -- | Pseudo-labeling (unstable) |
| v109 | ConvNeXtV2-T | 224 | 87.22 | -- | -- | -- | -- | -- | CAWR scheduler |
| v110 | ConvNeXtV2-T | 224 | 87.18 | -- | -- | -- | -- | -- | Larger batch (128) |
| v114 | ConvNeXtV2-T + Focal | 224 | 87.18 | -- | -- | -- | -- | -- | Focal loss only (no res increase) |
| v108 | ConvNeXtV2-B | 224 | 87.05 | -- | -- | -- | -- | -- | ConvNeXtV2-Base + weak aug |
| v103 | ConvNeXtV2-T | 224 | 85.55 | 84.64 | 86.26 | 85.31 | 84.70 | 85.29 | CAWR, born-again from v101 |
| v104 | ConvNeXtV2-B | 224 | 85.11 | 84.77 | 86.64 | 85.31 | 83.99 | 85.16 | Base model, born-again from v100 |
| v105 | ConvNeXtV2-T | 224 | 85.25 | -- | -- | -- | -- | -- | Over-regularized |
| v101 | ConvNeXtV2-T | 224 | 84.67 | 84.06 | 85.69 | 85.48 | 83.89 | 84.76 | Born-again from v100 |
| v102 | ConvNeXtV2-T | 224 | 84.64 | 84.06 | -- | -- | -- | -- | LR too low |
| v100 | ConvNeXtV2-T | 224 | 80.67 | 80.64 | 80.87 | 81.24 | 79.61 | 80.61 | First 5-fold baseline |
| v115 | ConvNeXtV2-T (6ch) | 224 | 75.86 | -- | -- | -- | -- | -- | FFT channels (from scratch) |
| v112 | EfficientNetV2-S | 224 | 76.94 | -- | -- | -- | -- | -- | EfficientNet (poor for radar) |

## Submission Results

| Submission File | Public Score | Private Score | Description |
|----------------|-------------|--------------|-------------|
| `submit1_best8_geo.csv` | **0.90008** | 0.89845 | 8-model geometric mean |
| `submit1_best5_plus_v38_geo.csv` | 0.89987 | **0.89906** | 6-model geo (v11+vit_v9+v18+v25+v24+v38) |
| `consensus_3plus.csv` | 0.89987 | 0.89906 | 6-model geo + consensus filtering (3+ agreement) |
| `submit2_best5_plus_v38_v33_geo.csv` | 0.90008 | 0.89845 | 7-model geo (adds v33) |

All four submission CSVs are included in this repository.

## Key Findings

1. **Weak augmentation > Strong augmentation**: Reducing augmentation strength (v106) was a breakthrough, narrowing the train-val gap significantly.

2. **Born-Again distillation works**: Iterative self-distillation consistently improved models by 1-2% per generation (v63 -> v65 -> v66 -> v67).

3. **Focal Loss helps hard examples**: Combined with higher resolution, focal loss (gamma=2) helped distinguish confusable sign pairs (e.g., M-N, U-V).

4. **Ensemble diversity is critical**: The best test score came from combining architecturally diverse models (ConvNeXtV2 + EfficientNetV2 + CAFormer) at multiple resolutions, not from ensembling the same model trained multiple times.

5. **Local validation != test performance**: Single-fold validation overestimated generalization; 5-fold CV was more reliable but still imperfect. TTA consistently helped test scores.

## How to Reproduce

1. Install dependencies: `pip install torch torchvision timm numpy pandas scikit-learn`
2. Train individual models: `python train_v11.py` (set `CUDA_VISIBLE_DEVICES` as needed)
3. Generate ensemble: `python ensemble.py` (requires trained checkpoints with `eval_probs_tta.pt`)

## Citation

```bibtex
@inproceedings{maitra2026msbad,
  title     = {{MSBAD} at {SignEval} 2026: Multi-Scale Born-Again Distillation Ensemble for Radar-Based Sign Language Recognition},
  author    = {Maitra, Arkadip and Patra, Suvajit and Samanta, Soumitra},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2026}
}
```

Thank you for using this repository. For any questions or support, please open an issue in this repository.
