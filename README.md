# dsp-dgcca-dehazing

Minimal code release for **Geometry-Aware Physical Dehazing with Depth-Guided Cross-Modal Channel Attention** (*Digital Signal Processing*, 2026, Article 106564, [DOI: 10.1016/j.dsp.2026.106564](https://doi.org/10.1016/j.dsp.2026.106564); manuscript DSP-D-26-03630).

This repo keeps only **model + train/eval** scripts (no figure / sensitivity / analysis tooling).

## Layout

```
models/          # FFA-Net + DG-CCA (fusion.py)
utils/           # datasets & losses
train_ots_v2_tv.py      # final training (SkyTV)
train_ots.py            # stage training
train_baseline_ots.py   # baseline
run_ablation.py         # multi-dataset evaluation
requirements.txt
```

## Environment

- Python 3.9 (conda env recommended)
- PyTorch 2.5.x + CUDA 12.1 (as used in our experiments)
- See `requirements.txt`

## Weights

Pretrained checkpoints are attached to the GitHub **Releases** (not in git history):

| File | Role |
|------|------|
| `ffa_dgcca_ots_dav2_epoch_120.pth` | Final DG-CCA + DAV2 depth model |
| `ffa_baseline_ots_epoch_100.pth` | Baseline FFA-Net on OTS |

Download from the latest Release, then place under `checkpoints_ots/`.

## Data

RESIDE / O-HAZE / NH-HAZE etc. are **not** bundled. Prepare local paths and (for the full model) DAV2 depth maps as in the paper. Point the dataset roots in the train/eval scripts or via their CLI/config flags.

## Citation

If you use this code, please cite:

```
Li, W., Cui, J., Ye, M., Chen, S., & Li, Q. (2026). Geometry-Aware Physical Dehazing with Depth-Guided Cross-Modal Channel Attention. Digital Signal Processing, Article 106564. https://doi.org/10.1016/j.dsp.2026.106564
```
