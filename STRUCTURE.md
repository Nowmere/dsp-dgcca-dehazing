# STRUCTURE — Paper 1 code release draft

Generated: 2026-10-11 (Asia/Shanghai)  
Source: `E:\Dehazing_Project\dsp-paper\code\`  
Destination: `E:\GrokBot-Workspace\04-code\dsp-dgcca-release\`  
Policy: **code-only**, no push, no secrets, no full checkpoint trees / datasets / latex.

## 1. Source survey (pre-copy)

### 1.1 `code\` directories (source sizes)

| Path | Approx. size | Notes |
|------|--------------|-------|
| `models/` | 0.033 MB | `fusion.py` (DG-CCA), `ffanet.py`, `ffanet_baseline.py` |
| `utils/` | 0.037 MB | datasets + losses |
| `scripts/` | 0.033 MB | depth gen / official weight checks |
| `depth_scripts/` | 0.025 MB | DAV2 generation / diagnostics |
| `figure_scripts/` | 0.374 MB | paper figure generators (many fig2 drafts) |
| root `*.py` / `*.json` | ~0.1 MB | train / ablation / sensitivity / tests |

No `.pth` / `.zip` / `.pdf` lived under `code\` itself. No `__pycache__` present at copy time. Secret-pattern scan on `code\` text files: **no API keys / credentials hits**.

### 1.2 `weights\` (NOT copied — document only)

| Dir | Approx. total | Typical file size |
|-----|---------------|-------------------|
| `checkpoints/` | 43.71 MB | ~2.9 MB / file (+ empty `dav2_hub/` on disk) |
| `checkpoints_baseline/` | 27.69 MB | ~2.8 MB / file |
| `checkpoints_baseline_ext/` | 55.4 MB | ~2.8 MB / file |
| `checkpoints_ots/` | 697.56 MB | ~2.8–8.8 MB / file |

Paper-facing OTS names (examples):

- `ffa_baseline_ots_epoch_100.pth` (~2.77 MB)
- `ffa_dgcca_ots_epoch_100.pth` (~2.91 MB)
- `ffa_dgcca_ots_dav2_epoch_100.pth` (~8.76 MB)
- `ffa_dgcca_ots_dav2_epoch_120.pth` (~8.76 MB) — final
- `ffa_dgcca_ots_dav2_adaptive_epoch_120.pth` (~8.77 MB)

Although individual demo weights are **<50MB**, this release still **excludes all `.pth`** by policy (avoid shipping a full training tree; host later via Drive/HF/Release assets).

## 2. Copied into release

- All of `models/`, `utils/`, `scripts/`, `depth_scripts/`, `figure_scripts/`
- Root training / eval / sensitivity scripts:
  - `train_ots_v2_tv.py`, `train_ots.py`, `train_baseline_ots.py`
  - `run_ablation.py`, `run_sensitivity_*.py`, `run_restormer_test.py`
  - `test_densehaze.py`, `test_densehaze_baseline.py`
  - `analyze_ihaze.py`, `create_mini_ots.py`
  - `sensitivity_v3_mini.json`, `sensitivity_v3_mini_lambda.json` (small logs)
- New: `README.md`, `STRUCTURE.md`, `requirements.txt`, `.gitignore`
- Empty placeholders: `data/.gitkeep`, `checkpoints_ots/.gitkeep`, `weights/.gitkeep`

Release tree after copy (before git objects): **87 files**, **~0.564 MB**.

## 3. Excluded (and why)

| Excluded | Why |
|----------|-----|
| `dsp-paper/weights/**/*.pth` | Full checkpoint trees; host externally |
| `dsp-paper/results/` | Large experiment outputs |
| `DSP-D-26-03630_*.zip`, `*.pptx`, large `*.pdf` | Submission / funding packages |
| `archive_dev/`, `Dehazing_Project_legacy/`, `submission_package/` bulk | Dev archives / EM packages |
| `paper_jvcir/`, latex sources, `main.pdf` | Default **code-only** |
| `baselines/` third-party methods | Separate codebases; not this method |
| Datasets under `data/` | Licensing + size |
| `.credentials`, API keys | Secrets |
| `test_correct_block.py` (only in `archive_dev`) | Archive helper; a couple of scripts still import it (see caveats) |

## 4. Entrypoint map (certainty)

| Script | Role | Certainty |
|--------|------|-----------|
| `train_ots_v2_tv.py` | Final train (SkyTV), saves `ffa_dgcca_ots_dav2_epoch_*.pth` | **High** (matches memory + save names) |
| `models/fusion.py` | DG-CCA QKV authority | **High** |
| `run_ablation.py` | Main multi-dataset ablation eval | **High** |
| `train_ots.py` / `train_baseline_ots.py` | Stage-1 / baseline | **High** |
| `utils/loss_v3_sky_tv_adaptive.py` | Adaptive SkyTV class present | **Medium** — not default-imported by `train_ots_v2_tv.py` (uses v2) |
| `analyze_ihaze.py` | I-HAZE depth-std analysis | **Low** for out-of-box use (hard-coded `E:\Dehazing_Project` paths) |
| `run_restormer_test.py` | Baseline Restormer comparison helper | **Medium** — may need Restormer code elsewhere |
| `figure_scripts/*` | Paper figures only | Optional |

## 5. Open questions (for user / parent)

1. **Public vs private** GitHub repo?
2. Proposed default name: `geometry-aware-physical-dehazing` vs `dsp-dgcca-dehazing`?
3. **Include LaTeX** later or keep code-only?
4. Which checkpoint(s) to host on Drive / Hugging Face / GitHub Release?
   - Minimal: `ffa_dgcca_ots_dav2_epoch_120.pth` (+ optional baseline `ffa_baseline_ots_epoch_100.pth`)
5. Add `LICENSE` (e.g. MIT / Apache-2.0 / research-only)?
6. Sanitize train scripts before public push (`RESUME_TRAINING=True` defaults; Chinese emoji logs; `hf-mirror` endpoint)?

## 6. Git status

Local `git init` + initial commit on `main` only. **No remote. No push.**