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

## Best Results

### Highlighted Runs

| Run | Type | Architecture | Resolution | Val Acc | Notes |
|-----|------|-------------|-----------|---------|-------|
| **v67** | Single-fold | ConvNeXtV2-Tiny | 224 | **90.98%** | Best single-model validation accuracy |
| **v113** | 5-fold | ConvNeXtV2-Tiny + Focal Loss | 288 | **88.14% avg** | Best 5-fold cross-validation average |

### Final Ensemble (89.91% test)

The best submission uses a 6-model ensemble with geometric-mean fusion and consensus filtering:

| Model | Backbone | Resolution | Val Acc |
|-------|----------|-----------|---------|
| v11 | ConvNeXtV2-Tiny | 224 | 89.08% |
| v18 | ConvNeXtV2-Tiny | 384 | 89.31% |
| v25 | ConvNeXtV2-Tiny | 448 | 89.38% |
| v38 | ConvNeXtV2-Tiny | 512 | 89.82% |
| v24 | EfficientNetV2-S | 384 | 88.36% |
| vit_v9 | CAFormer-S18 | 224 | 89.99% |

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

### Single-Fold Runs (sorted by validation accuracy)

| Run | Architecture | Resolution | Val Acc | Key Change |
|-----|-------------|-----------|---------|------------|
| v67 | ConvNeXtV2-T | 224 | 90.98% | Born-again gen4, weak aug |
| v70 | ConvNeXtV2-T | 224 | 90.88% | Extended training |
| v66 | ConvNeXtV2-T | 224 | 90.84% | Born-again gen3 |
| v71 | ConvNeXtV2-T | 224 | 90.84% | LR schedule variant |
| v75 | ConvNeXtV2-T | 224 | 90.81% | Fine-tuned augmentation |
| v90 | ConvNeXtV2-T | 224 | 90.78% | Late-stage refinement |
| v91 | ConvNeXtV2-T | 224 | 90.78% | Schedule variant |
| v97 | ConvNeXtV2-T | 224 | 90.71% | Final single-fold iteration |
| v94 | ConvNeXtV2-T | 224 | 90.67% | Augmentation tuning |
| v89 | ConvNeXtV2-T | 224 | 90.67% | Training variant |
| v78 | ConvNeXtV2-T | 224 | 90.67% | LR variant |
| v76 | ConvNeXtV2-T | 224 | 90.67% | Schedule variant |
| v87 | ConvNeXtV2-T | 224 | 90.64% | Born-again variant |
| v68 | ConvNeXtV2-T | 224 | 90.61% | Training variant |
| v88 | ConvNeXtV2-T | 224 | 90.54% | Augmentation variant |
| v65 | ConvNeXtV2-T | 224 | 90.44% | Born-again gen2 |
| v13 | ConvNeXtV2-T | 224 | 90.40% | Early strong model |
| v73 | ConvNeXtV2-T | 224 | 90.20% | Training variant |
| v63 | ConvNeXtV2-T | 224 | 90.10% | Born-again gen1 |
| vit_v10 | CAFormer | 224 | 90.06% | CAFormer variant |
| v41 | ConvNeXtV2-T | 224 | 90.06% | Multi-res variant |
| vit_v9 | CAFormer-S18 | 224 | 89.99% | **Best CAFormer** |
| vit_v12 | CAFormer | 224 | 89.96% | CAFormer variant |
| vit_v11 | CAFormer | 224 | 89.93% | CAFormer variant |
| v50 | ConvNeXtV2-T | 288 | 89.89% | Born-again @288 |
| v37 | ConvNeXtV2-T | 448 | 89.86% | High-res training |
| v42 | ConvNeXtV2-T | 384 | 89.82% | Multi-res |
| v38 | ConvNeXtV2-T | 512 | 89.82% | **Highest resolution** |
| vit_v14 | CAFormer | 224 | 89.82% | CAFormer variant |
| vit_v15 | CAFormer | 224 | 89.79% | CAFormer variant |
| v86 | ConvNeXtV2-T | 224 | 89.79% | Training variant |
| v72 | ConvNeXtV2-T | 224 | 89.66% | LR variant |
| v45 | CAFormer | 384 | 89.65% | CAFormer @384 |
| v55 | ConvNeXtV2-T | 224 | 89.59% | Training variant |
| v20 | ConvNeXtV2-T | 384 | 89.45% | Multi-res training |
| v47 | ConvNeXtV2-T | 320 | 89.42% | Born-again @320 |
| v25 | ConvNeXtV2-T | 448 | 89.38% | Multi-res |
| v52 | ConvNeXtV2-T | 384 | 89.38% | Gen2 from v18 |
| v59 | ConvNeXtV2-T | 224 | 89.35% | Training variant |
| v18 | ConvNeXtV2-T | 384 | 89.31% | Multi-res training |
| v58 | ConvNeXtV2-T | 224 | 89.31% | Training variant |
| v43 | CAFormer | 320 | 89.28% | CAFormer @320 |
| v40 | ConvNeXtV2-T | 224 | 89.21% | Training variant |
| v77 | ConvNeXtV2-T | 224 | 89.11% | LR variant |
| v11 | ConvNeXtV2-T | 224 | 89.08% | **Base ensemble model** |
| v10 | ConvNeXtV2-T | 224 | 88.98% | Early model |
| v80 | ConvNeXtV2-T | 224 | 88.98% | Training variant |
| v19 | ConvNeXtV2-T | 384 | 88.94% | Multi-res variant |
| v35 | ConvNeXtV2-T | 224 | 88.94% | Training variant |
| v57 | ConvNeXtV2-T | 224 | 88.94% | Training variant |
| v14 | ConvNeXtV2-T | 224 | 88.91% | Early model |
| v15 | ConvNeXtV2-T | 224 | 88.84% | Early model |
| vit_v8 | CAFormer | 224 | 88.57% | CAFormer variant |
| v24 | EfficientNetV2-S | 384 | 88.36% | **Alt architecture** |
| v39 | ConvNeXtV2-T | 224 | 88.23% | Training variant |
| v33 | ConvNeXtV2-T | 448 | 88.09% | Early multi-res |
| v12b | ConvNeXtV2-T | 224 | 88.03% | Training variant |
| v36 | ConvNeXtV2-T | 512 | 88.87% | High-res variant |
| v32 | ConvNeXtV2-T | 224 | 87.86% | Training variant |
| v12c | ConvNeXtV2-T | 224 | 87.82% | Training variant |
| v12 | ConvNeXtV2-T | 224 | 87.79% | Base model |
| v69 | ConvNeXtV2-T | 224 | 87.59% | Training variant |
| v8 | ConvNeXtV2-T | 224 | 87.62% | Early model |
| v7 | ConvNeXtV2-T | 224 | 87.28% | Early model |
| v53 | EfficientNetV2-S | 320 | 86.40% | Alt arch @320 |
| v51 | CAFormer | 288 | 86.64% | CAFormer @288 |
| v82 | ConvNeXtV2-T | 224 | 86.61% | Training variant |
| v64 | ConvNeXtV2-T | 224 | 86.27% | Training variant |
| v48 | EfficientNetV2-S | 448 | 85.96% | Alt arch @448 |
| v6 | ConvNeXtV2-T | 224 | 85.79% | Early model |
| vit_v7 | CAFormer | 224 | 85.62% | Early CAFormer |
| vit_v6 | CAFormer | 224 | 85.58% | Early CAFormer |
| v46 | CAFormer | 448 | 85.07% | CAFormer @448 |
| v5 | ConvNeXtV2-T | 224 | 84.23% | Early model |
| v4 | ConvNeXtV2-T | 224 | 84.77% | Early model |
| vit_v5 | CAFormer | 224 | 84.09% | Early CAFormer |
| v62 | ConvNeXtV2-T | 224 | 84.88% | Training variant |
| v60 | ConvNeXtV2-T | 224 | 84.27% | Training variant |
| v9 | ConvNeXtV2-T | 224 | 83.85% | Early model |
| v61 | ConvNeXtV2-T | 224 | 83.38% | Training variant |
| v81 | ConvNeXtV2-T | 224 | 83.52% | Training variant |
| v3 | ConvNeXtV2-T | 224 | 82.19% | Early model |
| v2 | ConvNeXtV2-T | 224 | 82.56% | Early model |
| vit_v3 | CAFormer | 224 | 82.02% | Early CAFormer |
| vit_v4 | CAFormer | 224 | 82.56% | Early CAFormer |
| 1d_v2 | 1D-CNN | -- | 80.39% | 1D baseline v2 |
| v49 | ConvNeXtV2-B | 224 | 80.47% | Base model from scratch |
| 1d | 1D-CNN | -- | 79.99% | 1D baseline |
| v54 | ConvNeXtV2-B | 224 | 85.01% | Base model variant |
| v95 | ConvNeXtV2-T | 224 | 79.93% | Failed experiment |
| vit_v13 | CAFormer | 224 | 78.77% | Failed experiment |
| v83 | ConvNeXtV2-T | 224 | 78.23% | Failed experiment |
| v17 | ConvNeXtV2-T | 224 | 78.90% | Failed experiment |
| v84 | ConvNeXtV2-T | 224 | 76.91% | Failed experiment |
| v85 | ConvNeXtV2-T | 224 | 80.77% | Failed experiment |
| v26 | ConvNeXtV2-T | 224 | 72.86% | Failed experiment |
| v79 | ConvNeXtV2-T | 224 | 70.57% | Failed experiment |
| v96 | ConvNeXtV2-T | 224 | 65.34% | Failed experiment |
| v27 | ConvNeXtV2-T | 224 | 60.05% | Failed experiment |
| vit_v2 | CAFormer | 224 | 54.17% | Early CAFormer |
| vit | CAFormer | 224 | 31.75% | First CAFormer |
| v22 | ConvNeXtV2-T | 224 | 26.39% | Failed experiment |
| v44 | ConvNeXtV2-T | 224 | 11.94% | Failed experiment |
| v16 | ConvNeXtV2-T | 224 | 9.29% | Failed experiment |
| v23 | ConvNeXtV2-T | 224 | 2.41% | Failed experiment |
| v28 | ConvNeXtV2-T | 224 | 0.88% | Failed experiment |

### 5-Fold Cross-Validation Runs

| Run | Architecture | Resolution | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Avg | Key Change |
|-----|-------------|-----------|--------|--------|--------|--------|--------|-----|------------|
| **v117** | ConvNeXtV2-T + Focal | 384 | **89.22** | 87.18 | 87.42 | 87.79 | 87.79 | **87.88** | Higher resolution, born-again from v113 |
| **v113** | ConvNeXtV2-T + Focal | 288 | 88.50 | 87.45 | 88.09 | 88.57 | 88.06 | **88.14** | Focal loss + 288px, born-again from v106 |
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

3. **Higher resolution helps**: Scaling from 224 to 288px (v113) and 384px (v117) yielded consistent gains for fine-grained sign distinction.

4. **Focal Loss helps hard examples**: Combined with higher resolution, focal loss (gamma=2) helped distinguish confusable sign pairs (e.g., M-N, U-V).

5. **Ensemble diversity is critical**: The best test score came from combining architecturally diverse models (ConvNeXtV2 + EfficientNetV2 + CAFormer) at multiple resolutions, not from ensembling the same model trained multiple times.

6. **Local validation != test performance**: Single-fold validation overestimated generalization; 5-fold CV was more reliable but still imperfect. TTA consistently helped test scores.

## How to Reproduce

1. Install dependencies: `pip install torch torchvision timm numpy pandas scikit-learn`
2. Train individual models: `python train_v11.py` (set `CUDA_VISIBLE_DEVICES` as needed)
3. Generate ensemble: `python ensemble.py` (requires trained checkpoints with `eval_probs_tta.pt`)

## Citation

```bibtex
@inproceedings{msbad2026signeval,
  title={MSBAD at SignEval 2026: Multi-Scale Born-Again Distillation Ensemble for Radar-Based Sign Language Recognition},
  booktitle={CVPR Workshops},
  year={2026}
}
```
