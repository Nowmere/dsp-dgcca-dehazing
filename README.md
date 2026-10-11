# Geometry-Aware Physical Dehazing with DG-CCA

**中文 / English**

Code release draft for the paper:

> **Geometry-Aware Physical Dehazing with Depth-Guided Cross-Modal Channel Attention**  
> Wujin Li, Jianxiang Cui, Mingqiang Ye, Siyuan Chen, Qianming Li  
> *Digital Signal Processing* (Elsevier), Manuscript **DSP-D-26-03630** — **Accepted 2026-09-17**  
> Affiliation: Hunan Institute of Science and Technology, Yueyang, China  
> Corresponding author: Qianming Li (`liqianming@hnist.edu.cn`)

本仓库为论文方法的**代码整理稿**（本地 `git init`，尚未绑定远程 / 未 push）。  
This folder is a **code-only** release draft (local git only; no remote/push yet).

---

## 方法摘要 / Method (short)

- Frozen **Depth Anything V2 (Small)** as geometric prior (~24.18M params, no grad).
- **DG-CCA** (Depth-Guided Cross-modal Channel Attention): Q from RGB features, K/V from depth features; authoritative implementation: `models/fusion.py` (`ChannelCrossAttention`). Module params at C=64: **18,113**.
- **SkyTV** targeted sky TV loss (mask threshold **τ=0.1**; temperature symbol is **α**, not τ).
- Trainable dehazing backbone (FFA-style) ≈ **0.752M** trainable params.
- Final training entry used in the paper pipeline: `train_ots_v2_tv.py` (Stage 1 + SkyTV Stage 2 → `ffa_dgcca_ots_dav2_epoch_120.pth`).

数字红线以项目事实源为准；本 README 不替代 `memory.md`。  
Numeric claims should be cross-checked against the project single source of truth before citation.

---

## 环境 / Environment

Verified local conda env **`dehaze`**:

| Item | Value |
|------|--------|
| Python | 3.9.25 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| GPU (authors) | RTX 4060 Ti 16GB |

```bash
conda activate dehaze
pip install -r requirements.txt
# or install CUDA wheels first, then the rest of requirements.txt
```

Depth prior is loaded via Hugging Face `transformers` in eval/ablation scripts:

`depth-anything/Depth-Anything-V2-Small-hf`

（国内可继续使用脚本中已有的 `HF_ENDPOINT=https://hf-mirror.com`。）

---

## 仓库结构 / Layout

```
dsp-dgcca-release/
  models/           # FFA_DGCCA_Net, FFA_Baseline_Net, DG-CCA (fusion.py)
  utils/            # datasets + losses (incl. SkyTV / adaptive SkyTV)
  scripts/          # depth precompute helpers, weight checks
  depth_scripts/    # DAV2 depth generation / checks
  figure_scripts/   # paper figure generators (optional)
  train_ots_v2_tv.py      # ★ final train entry (SkyTV)
  train_ots.py            # Stage-1 style train (no SkyTV)
  train_baseline_ots.py   # baseline FFA without DG-CCA
  run_ablation.py         # ablation eval on SOTS / O/NH/I-HAZE / Dense-Haze
  run_sensitivity_*.py    # τ / λ sensitivity grids
  test_densehaze*.py      # Dense-Haze tests
  requirements.txt
  STRUCTURE.md            # what was copied vs excluded
  data/                   # empty placeholder — put RESIDE / HAZE sets here
  checkpoints_ots/        # empty — put .pth here locally
  weights/                # empty placeholder
```

完整说明见 `STRUCTURE.md`。

---

## 数据 / Datasets（未随仓库分发）

Scripts expect paths under the repo root, e.g.:

```
data/RESIDE/OTS/{hazy,clear,depth_dav2}/
data/RESIDE/SOTS/outdoor/{hazy,clear}/
data/O-HAZE/{hazy,GT}/
data/NH-HAZE/{hazy,GT}/
data/I-HAZE/{hazy,GT}/
data/Dense_Haze/{hazy,GT}/
```

Depth maps for training are typically `.pt` / `.npy` produced by `depth_scripts/` or `scripts/generate_*_depth.py`.

**NOT included:** RESIDE / O-HAZE / NH-HAZE / I-HAZE / Dense-Haze images, precomputed depth tensors, or baseline third-party code under `dsp-paper/baselines/`.

---

## 权重 / Checkpoints（未随仓库分发）

Do **not** commit `.pth` files. Local author copies live under:

`E:\Dehazing_Project\dsp-paper\weights\`

Key filenames expected by `run_ablation.py` / paper narrative:

| File (under `checkpoints_ots/`) | Role | Approx. size (local) |
|---------------------------------|------|----------------------|
| `ffa_baseline_ots_epoch_100.pth` | Ablation A (baseline) | ~2.8 MB |
| `ffa_dgcca_ots_epoch_100.pth` | Early DG-CCA | ~2.9 MB |
| `ffa_dgcca_ots_dav2_epoch_100.pth` | Stage-1 end | ~8.8 MB |
| `ffa_dgcca_ots_dav2_epoch_120.pth` | ★ Final (SkyTV stage) | ~8.8 MB |
| `ffa_dgcca_ots_dav2_adaptive_epoch_120.pth` | Adaptive SkyTV variant | ~8.8 MB |

Also: early `weights/checkpoints/*.pth` (~2.9 MB each) and optional DAV2 hub cache (download via HF / torch.hub).

Place copies into this repo's `checkpoints_ots/` for local eval, or set paths in scripts.  
Public hosting (GitHub Release / Google Drive / Hugging Face) is **TBD** — see open questions in `STRUCTURE.md`.

---

## 训练 / Training (high level)

1. Prepare OTS hazy/clear + DAV2 depths under `data/RESIDE/OTS/`.
2. Edit resume flags in the train script (**default `RESUME_TRAINING=True`** — set `False` for a fresh run).
3. Final recipe (paper pipeline):

```bash
python train_ots_v2_tv.py
# saves checkpoints_ots/ffa_dgcca_ots_dav2_epoch_{N}.pth
```

Related:

```bash
python train_ots.py              # without SkyTV (earlier stage style)
python train_baseline_ots.py     # FFA baseline
```

Hyperparams in scripts (typical): crop 256, AdamW 1e-4, cosine LR, AMP, batch 4 × accum 4 (DG-CCA) / larger batch for baseline.

---

## 评测 / Evaluation (high level)

```bash
# Place the four ablation checkpoints under checkpoints_ots/ first
python run_ablation.py

python test_densehaze.py
python run_sensitivity_v3_mini.py   # example sensitivity run
```

---

## 引用 / Citation (stub)

```bibtex
@article{li2026geometry,
  title   = {Geometry-Aware Physical Dehazing with Depth-Guided Cross-Modal Channel Attention},
  author  = {Li, Wujin and Cui, Jianxiang and Ye, Mingqiang and Chen, Siyuan and Li, Qianming},
  journal = {Digital Signal Processing},
  year    = {2026},
  note    = {Accepted. Manuscript DSP-D-26-03630. DOI/pages TBD.}
}
```

---

## License / Release status

- Remote GitHub repo name (proposed): `geometry-aware-physical-dehazing` or `dsp-dgcca-dehazing`
- Visibility (public / private): **not decided**
- LaTeX / PDF / revision packages: **not included** (code-only by default)
- License file: **not added yet**

---

## 已知注意点 / Known caveats

1. `train_ots_v2_tv.py` imports `utils.loss_v2_sky_tv.CompositeLoss_SkyTV`; `utils/loss_v3_sky_tv_adaptive.py` exists but is **not** wired as the default train entry — treat as experimental/variant unless you change the import.
2. `analyze_ihaze.py` and a few helper scripts hard-code older absolute paths / missing `test_correct_block` (lives only under `archive_dev`, excluded). Prefer `run_ablation.py` + `depth_scripts/`.
3. `figure_scripts/` contains many iterative figure drafts (`draw_fig2_v*`); keep for reproducibility of paper figures, not required for train/eval.
4. No API keys or credentials are shipped in this tree.