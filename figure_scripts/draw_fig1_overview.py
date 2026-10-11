# -*- coding: utf-8 -*-
"""
Figure 1 (overview) for JVCIR submission: challenges -> method -> innovations -> results.
Measurement-driven layout (boxes auto-sized to rendered text, no overflow).

Red-line compliance:
  - exactly ONE DG-CCA module, QKV cross-modal internals only (no GAP/MLP/Sigmoid);
  - alpha = learnable temperature, tau = 0.1 mask threshold, mask = \\mathcal{M};
  - F_out = W_p(A V) + F_rgb; 18,113 params; DAV2 24.18 M frozen;
  - Stage 2: backbone kept trainable.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

fig, ax = plt.subplots(figsize=(19.6, 8.8), dpi=110)
ax.set_xlim(0, 1960)
ax.set_ylim(0, 880)
ax.axis("off")
fig.patch.set_facecolor("white")
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = ax.transData.inverted()

# ---- palette (identical to fig2 v2) -----------------------------------------
C_RGB,  F_RGB  = "#3B6EA5", "#E8F0FE"
C_DEP,  F_DEP  = "#B26B2B", "#FBE5D6"
C_FRZ,  F_FRZ  = "#A34A45", "#FCE8E6"
C_DGC,  F_DGC  = "#B45F06", "#FDF2E3"
C_ATT,  F_ATT  = "#6A329F", "#F3E8FD"
C_OUT,  F_OUT  = "#2E7D32", "#E6F4EA"
C_GRAY         = "#777777"
F_GRAY         = "#EFEFEF"
INK   = "#1a1a2e"
GRAY  = "#666666"
LIGHT = "#D9D9D9"

# ---- helpers -----------------------------------------------------------------
_mcache = {}

def measure(text, fs, bold=False):
    key = (text, fs, bold)
    if key in _mcache:
        return _mcache[key]
    t = ax.text(-3000, -3000, text, fontsize=fs, family="DejaVu Sans",
                weight="bold" if bold else "normal", linespacing=1.35,
                ha="center", va="center")
    bb = t.get_window_extent(renderer).transformed(inv)
    w, h = bb.width, bb.height
    t.remove()
    _mcache[key] = (w, h)
    return w, h


def mbox(cx, cy, text, fc, ec, fs=8.2, bold=False, pad_x=26, pad_y=20,
         min_w=0, min_h=0, ls="-", lw=1.5, tc=INK, rsize=7):
    w_, h_ = measure(text, fs, bold)
    w = max(w_ + pad_x, min_w)
    h = max(h_ + pad_y, min_h)
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle=f"round,pad=0.02,rounding_size={rsize}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", family="DejaVu Sans",
            color=tc, linespacing=1.35)
    return dict(cx=cx, cy=cy, w=w, h=h,
                x0=cx - w / 2, x1=cx + w / 2, y0=cy - h / 2, y1=cy + h / 2)


def fbox(x0, y0, w, h, fc, ec, lw=1.6, ls="-", rsize=8):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                 boxstyle=f"round,pad=0.02,rounding_size={rsize}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls))


def arr(x1, y1, x2, y2, color="#555555", lw=1.6, ls="-", ms=14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=ms, linewidth=lw, color=color, linestyle=ls))


def seg(pts, color="#555555", lw=1.6, ls="-"):
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            color=color, lw=lw, linestyle=ls, solid_capstyle="round")


def badge(cx, cy, r, label, color, fs=8.5):
    ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor="white", linewidth=1.3))
    ax.text(cx, cy - 0.5, label, ha="center", va="center", fontsize=fs,
            color="white", weight="bold", family="DejaVu Sans")


def tag(cx, cy, text, color):
    w_, _ = measure(text, 6.3, True)
    w = w_ + 14
    h = 16
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=4",
                 facecolor=color, edgecolor="none"))
    ax.text(cx, cy - 0.4, text, ha="center", va="center", fontsize=6.3,
            weight="bold", family="DejaVu Sans", color="white")


# ---- real RESIDE-OTS thumbnails ----------------------------------------------
HAZY_PATH = r"E:\Dehazing_Project\data\RESIDE\OTS\hazy\0025_0.8_0.2.jpg"
CLEAR_PATH = r"E:\Dehazing_Project\data\RESIDE\OTS\clear\0025.jpg"
_img_cache = {}

def load_crop(path, aspect):
    if path in _img_cache:
        img = _img_cache[path]
    else:
        img = mpimg.imread(path)
        _img_cache[path] = img
    h, w = img.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        img = img[:, x0:x0 + nw]
    else:
        nh = int(round(w / aspect))
        y0 = (h - nh) // 2
        img = img[y0:y0 + nh, :]
    return img


def thumb(x0, y0, w, h, path):
    ax.imshow(load_crop(path, w / h), extent=(x0, x0 + w, y0, y0 + h),
              interpolation="bilinear")
    frame = C_RGB if "hazy" in path.lower() else C_OUT
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=frame, linewidth=1.6))


# ===============================================================================
# panel titles
# ===============================================================================
ax.text(30, 852, "(a) Challenges", fontsize=10.5, weight="bold",
        family="DejaVu Sans", color=INK)
ax.text(390, 852, "(b) Proposed method — two streams, one DG-CCA module, two-stage curriculum",
        fontsize=10.5, weight="bold", family="DejaVu Sans", color=INK)
ax.text(1460, 852, "(c) Core innovations", fontsize=10.5, weight="bold",
        family="DejaVu Sans", color=INK)
ax.text(30, 252, "(d) Key results", fontsize=10.5, weight="bold",
        family="DejaVu Sans", color=INK)

# ===============================================================================
# (b) method — RGB stream  (centerline y=740)
# ===============================================================================
RGB_CY = 740
TH_W, TH_H = 90, 84
thumb(380, RGB_CY - TH_H / 2, TH_W, TH_H, HAZY_PATH)
ax.text(425, RGB_CY + TH_H / 2 + 32, "Hazy Image", fontsize=8.4, weight="bold",
        family="DejaVu Sans", color=INK, ha="center")
ax.text(425, RGB_CY + TH_H / 2 + 16, "$I\\in\\mathbb{R}^{3\\times H\\times W}$",
        fontsize=7.4, family="DejaVu Sans", color=GRAY, ha="center")

shallow_txt = "Shallow Conv (3$\\times$3)\nshallow features"
ffa_txt = "FFA-Net Group Blocks\nmulti-scale feature\nextraction & reconstruction"
dgc_l1 = "DG-CCA  $\\times$1"
dgc_l2 = "Depth-Guided Cross-modal\nChannel Attention"
dgc_l3 = "$Q\\!\\leftarrow\\!F_{rgb}$;  $K,V\\!\\leftarrow\\!F_{d}$"

w_sh = measure(shallow_txt, 8.0, True)[0] + 26
w_ff = measure(ffa_txt, 7.5, True)[0] + 26
w_dg = max(measure(dgc_l1, 10, True)[0],
           measure(dgc_l2, 7.6, False)[0],
           measure(dgc_l3, 7.0, False)[0]) + 28
w_chip = measure("$F_{rgb}$", 10, True)[0] + 24

X0 = 380 + TH_W
X_END = 1445
total = w_sh + w_chip + w_dg + w_chip + w_ff
g = (X_END - TH_W - X0 - total) / 6.0

x = X0 + g
sh_cx = x + w_sh / 2; x += w_sh + g
chip1_cx = x + w_chip / 2; x += w_chip + g
dgc_cx = x + w_dg / 2; x += w_dg + g
chip2_cx = x + w_chip / 2; x += w_chip + g
ffa_cx = x + w_ff / 2; x += w_ff + g
out_x0 = x

shallow = mbox(sh_cx, RGB_CY, shallow_txt, F_RGB, C_RGB, fs=8.0, bold=True, min_h=66)
chip1 = mbox(chip1_cx, RGB_CY, "$F_{rgb}$", F_RGB, C_RGB, fs=10, bold=True,
             min_w=w_chip, min_h=40)
ax.text(chip1_cx, chip1["y0"] - 11, "$C=64$", fontsize=6.8, family="DejaVu Sans",
        color=C_RGB, ha="center")

# DG-CCA — the single module
h_dg = 112
fbox(dgc_cx - w_dg / 2, RGB_CY - h_dg / 2, w_dg, h_dg, F_DGC, C_DGC, lw=2.0)
ax.text(dgc_cx, RGB_CY + 38, dgc_l1, fontsize=10, weight="bold",
        family="DejaVu Sans", color=C_DGC, ha="center")
ax.text(dgc_cx, RGB_CY + 8, dgc_l2, fontsize=7.6, family="DejaVu Sans",
        color="#7a4a06", ha="center", linespacing=1.3)
ax.text(dgc_cx, RGB_CY - 40, dgc_l3, fontsize=7.0, family="DejaVu Sans",
        color="#7a4a06", ha="center")
dgc = dict(cx=dgc_cx, cy=RGB_CY, w=w_dg, h=h_dg,
           x0=dgc_cx - w_dg / 2, x1=dgc_cx + w_dg / 2,
           y0=RGB_CY - h_dg / 2, y1=RGB_CY + h_dg / 2)
badge(dgc["x1"] - 2, dgc["y1"] + 2, 11, "1", C_ATT, fs=9)

chip2 = mbox(chip2_cx, RGB_CY, "$F_{out}$", F_OUT, C_OUT, fs=10, bold=True,
             min_w=w_chip, min_h=40)
ffa = mbox(ffa_cx, RGB_CY, ffa_txt, F_RGB, C_RGB, fs=7.5, bold=True, min_h=66)
ax.text(ffa_cx, ffa["y0"] - 13, "trainable", fontsize=7.2, family="DejaVu Sans",
        color=C_RGB, ha="center", style="italic")

thumb(out_x0, RGB_CY - TH_H / 2, TH_W, TH_H, CLEAR_PATH)
ax.text(out_x0 + TH_W / 2, RGB_CY - TH_H / 2 - 14, "Dehazed Output", fontsize=8.4,
        weight="bold", family="DejaVu Sans", color=INK, ha="center")
ax.text(out_x0 + TH_W / 2, RGB_CY - TH_H / 2 - 30, "$\\hat{I}$", fontsize=7.8,
        family="DejaVu Sans", color=C_OUT, ha="center")

arr(380 + TH_W + 2, RGB_CY, shallow["x0"] - 2, RGB_CY, color=C_RGB)
arr(shallow["x1"] + 2, RGB_CY, chip1["x0"] - 2, RGB_CY, color=C_RGB)
arr(chip1["x1"] + 2, RGB_CY, dgc["x0"] - 2, RGB_CY, color=C_RGB)
arr(dgc["x1"] + 2, RGB_CY, chip2["x0"] - 2, RGB_CY, color=C_RGB)
arr(chip2["x1"] + 2, RGB_CY, ffa["x0"] - 2, RGB_CY, color=C_RGB)
arr(ffa["x1"] + 2, RGB_CY, out_x0 - 3, RGB_CY, color=C_RGB)

# ===============================================================================
# (b) method — depth stream  (centerline y=520)
# ===============================================================================
DEP_CY = 520
dav2_txt = "Frozen DAV2 (small)\n24.18 M · no gradients"
dmap_txt = "Depth Map\n$D\\in\\mathbb{R}^{1\\times H\\times W}$\nmin-max $\\rightarrow [0,1]$"
sky_txt = "Soft Sky Suppression\n$D\\cdot\\mathrm{clip}(D/\\tau,\\,0,\\,1)$\n$\\tau=0.1$, attenuates sky noise"
adp_txt = "Depth Adapter\n3$\\times$3 conv (C/2) + ReLU\n3$\\times$3 conv $\\rightarrow$ C"

w_dav2 = measure(dav2_txt, 8.0, True)[0] + 26
w_dmap = measure(dmap_txt, 8.0, True)[0] + 26
w_sky = measure(sky_txt, 7.8, True)[0] + 26
w_adp = measure(adp_txt, 7.8, True)[0] + 26

DX0 = 380
dg = 24
dav2_cx = DX0 + w_dav2 / 2
dmap_cx = DX0 + w_dav2 + dg + w_dmap / 2
sky_cx = DX0 + w_dav2 + w_dmap + 2 * dg + w_sky / 2
adp_cx = DX0 + w_dav2 + w_dmap + w_sky + 3 * dg + w_adp / 2

dav2 = mbox(dav2_cx, DEP_CY, dav2_txt, F_GRAY, C_GRAY, fs=8.0, bold=True, min_h=64)
tag(dav2["x1"] - 24, dav2["y1"], "FROZEN", C_FRZ)
dmap = mbox(dmap_cx, DEP_CY, dmap_txt, F_DEP, C_DEP, fs=8.0, bold=True, min_h=64)
sky = mbox(sky_cx, DEP_CY, sky_txt, F_DEP, C_DEP, fs=7.8, bold=True, min_h=64,
           ls=(0, (4, 3)))
badge(sky["x1"] - 2, sky["y1"] + 2, 11, "2", C_FRZ, fs=9)
adp = mbox(adp_cx, DEP_CY, adp_txt, F_DEP, C_DEP, fs=7.8, bold=True, min_h=64)

depth_ymin = min(dav2["y0"], dmap["y0"], sky["y0"], adp["y0"])
ax.text((dav2_cx + adp_cx) / 2, depth_ymin - 26,
        "depth prior branch — frozen encoder, trainable adapter",
        fontsize=7.6, family="DejaVu Sans", color="#8a5a2b", ha="center", style="italic")

# input branch: down from the hazy thumbnail into DAV2
in_x = 425
seg([(in_x, RGB_CY - TH_H / 2 - 2), (in_x, dav2["y1"] + 6)], color=C_FRZ, lw=1.5)
arr(in_x, dav2["y1"] + 6, in_x, dav2["y1"] + 2, color=C_FRZ, lw=1.5, ms=12)
ax.text(in_x - 14, 630, "same input $I$", fontsize=7.0, family="DejaVu Sans",
        color=GRAY, ha="center", va="center", rotation=90)

arr(dav2["x1"] + 2, DEP_CY, dmap["x0"] - 2, DEP_CY, color=C_DEP)
arr(dmap["x1"] + 2, DEP_CY, sky["x0"] - 2, DEP_CY, color=C_DEP)
arr(sky["x1"] + 2, DEP_CY, adp["x0"] - 2, DEP_CY, color=C_DEP)

# adapter elbow up into DG-CCA (K/V inputs)
elb_y = 630
seg([(adp_cx, adp["y1"] + 2), (adp_cx, elb_y), (dgc_cx, elb_y)], color=C_DEP, lw=1.7)
arr(dgc_cx, elb_y, dgc_cx, dgc["y0"] - 3, color=C_DEP, lw=1.7)
ax.text((adp_cx + dgc_cx) / 2, elb_y + 12,
        "$F_{d}\\in\\mathbb{R}^{C\\times H\\times W}$  $\\rightarrow$  K, V of DG-CCA",
        fontsize=7.6, family="DejaVu Sans", color=C_DEP, ha="center")

# ===============================================================================
# (b) two-stage training bar  (y 310..430)
# ===============================================================================
st1_txt = ("Stage 1 · global training (epochs 1–100)\n"
           "$\\mathcal{L}_{rec}=\\mathcal{L}_1+0.05\\cdot\\mathcal{L}_{edge}$\n"
           "whole network trainable · DG-CCA fully active")
st2_txt = ("Stage 2 · regional decoupled fine-tuning (epochs 101–120)\n"
           "$\\mathcal{L}_{total}=\\mathcal{L}_{rec}+0.01\\cdot\\mathcal{L}_{TV}$   (backbone kept trainable)\n"
           "Sky TV Loss on mask $\\mathcal{M}$ only:  $\\tau=0.1$, 3$\\times$3 opening, $\\lambda=0.01$")

w_s1 = measure(st1_txt, 7.9, True)[0] + 28
w_s2 = measure(st2_txt, 7.9, True)[0] + 28
pair = w_s1 + w_s2 + 46
sx0 = 380 + (1065 - pair) / 2
st1 = mbox(sx0 + w_s1 / 2, 370, st1_txt, "#FFFFFF", C_RGB, fs=7.9, bold=True,
           pad_x=28, pad_y=20, min_h=110)
st2 = mbox(sx0 + w_s1 + 46 + w_s2 / 2, 370, st2_txt, "#FFFFFF", C_DGC, fs=7.9,
           bold=True, pad_x=28, pad_y=20, min_h=110)
arr(st1["x1"] + 3, 370, st2["x0"] - 3, 370, color=GRAY, lw=1.8, ms=16)
ax.text(st1["cx"], st1["y0"] - 16, "two-stage training curriculum",
        fontsize=7.8, family="DejaVu Sans", color=GRAY, ha="center", style="italic")

# ===============================================================================
# (a) challenges
# ===============================================================================
def challenge(y0, y1, num, title, body):
    fbox(30, y0, 300, y1 - y0, "#FFFFFF", "#C75146", lw=1.5)
    badge(54, y1 - 24, 12, num, "#C75146", fs=9.5)
    ax.text(74, y1 - 24, title, fontsize=8.8, weight="bold", family="DejaVu Sans",
            color=INK, va="center")
    ax.text(30 + 300 / 2, (y0 + y1) / 2 - 16, body, fontsize=7.6, family="DejaVu Sans",
            color="#444444", ha="center", va="center", linespacing=1.4)

challenge(680, 830, "1", "No depth cues in 2D dehazing",
          "a pure 2D net cannot distinguish a white\n"
          "wall at 2 m from a fog layer at 100 m")
challenge(500, 650, "2", "Infinity-depth noise in sky",
          "monocular DAV2 outputs chaotic depth at\n"
          "optical infinity — noisy sky depth corrupts\n"
          "the depth prior")
challenge(320, 470, "3", "Large SOTA models impractical",
          "depth-guided competitors are much larger\n"
          "(e.g., DNMGDT $\\approx$3$\\times$) — our trainable\n"
          "model has only 0.752 M params")

# ===============================================================================
# (c) innovations
# ===============================================================================
# innovation 1 — DG-CCA
fbox(1460, 640, 470, 190, F_ATT, C_ATT, lw=1.8)
badge(1486, 806, 12, "1", C_ATT, fs=9.5)
ax.text(1506, 806, "DG-CCA — cross-modal channel attention", fontsize=8.8,
        weight="bold", family="DejaVu Sans", color=C_ATT, va="center")
ax.text(1695, 752, "Q from RGB features;  K, V from depth features", fontsize=7.8,
        family="DejaVu Sans", color=INK, ha="center")
ax.text(1695, 726, "$A=\\mathrm{softmax}(\\alpha\\,\\hat{Q}\\hat{K}^{\\top})\\in\\mathbb{R}^{C\\times C}$"
        "   ($\\alpha$: learnable temperature)", fontsize=7.8, family="DejaVu Sans",
        color=INK, ha="center")
ax.text(1695, 700, "$F_{out}=W_{p}(A\\,V)+F_{rgb}$  (1$\\times$1 projection + residual)",
        fontsize=7.8, family="DejaVu Sans", color=INK, ha="center")
ax.text(1695, 666, "single module · 18,113 params · $\\approx$5 % of backbone · C=64",
        fontsize=7.4, family="DejaVu Sans", color=C_ATT, ha="center", style="italic")

# innovation 2 — SkyTV
fbox(1460, 420, 470, 190, F_FRZ, C_FRZ, lw=1.8)
badge(1486, 586, 12, "2", C_FRZ, fs=9.5)
ax.text(1506, 586, "SkyTV — taming infinity-depth noise", fontsize=8.8,
        weight="bold", family="DejaVu Sans", color=C_FRZ, va="center")
ax.text(1695, 534, "forward:  soft sky suppression  $D\\cdot\\mathrm{clip}(D/\\tau,\\,0,\\,1)$",
        fontsize=7.8, family="DejaVu Sans", color=INK, ha="center")
ax.text(1695, 508, "fine-tune:  Targeted Sky TV Loss on mask $\\mathcal{M}$",
        fontsize=7.8, family="DejaVu Sans", color=INK, ha="center")
ax.text(1695, 482, "$\\tau=0.1$  ·  $\\lambda=0.01$  ·  3$\\times$3 morphological opening",
        fontsize=7.8, family="DejaVu Sans", color=INK, ha="center")
ax.text(1695, 448, "active only in the final 20 epochs (Stage 2)",
        fontsize=7.4, family="DejaVu Sans", color=C_FRZ, ha="center", style="italic")

# connector DG-CCA -> innovation 1 (through the empty band above the RGB stream)
seg([(dgc_cx, dgc["y1"] + 2), (dgc_cx, 826)], color=C_ATT, lw=1.2, ls=(0, (4, 3)))
arr(dgc_cx, 826, 1458, 826, color=C_ATT, lw=1.2, ls=(0, (4, 3)), ms=11)

# ===============================================================================
# (d) results — four stat cards
# ===============================================================================
def card(x0, x1, stat, stat_color, metric, cap1, cap2):
    fbox(x0, 48, x1 - x0, 180, "#FFFFFF", LIGHT, lw=1.3)
    ax.add_patch(Rectangle((x0 + 6, 218), x1 - x0 - 12, 6, facecolor=stat_color,
                           edgecolor="none"))
    cx = (x0 + x1) / 2
    ax.text(cx, 182, stat, fontsize=17, weight="bold", family="DejaVu Sans",
            color=stat_color, ha="center")
    ax.text(cx, 140, metric, fontsize=8.8, weight="bold", family="DejaVu Sans",
            color="#444444", ha="center")
    ax.text(cx, 108, cap1, fontsize=7.5, family="DejaVu Sans", color=GRAY, ha="center")
    ax.text(cx, 92, cap2, fontsize=7.5, family="DejaVu Sans", color=GRAY, ha="center")

card(30, 482, "+2.35 dB", C_RGB, "SOTS-Indoor · PSNR gain",
     "zero-shot cross-domain generalization", "(22.37 vs 20.02 dB, vs FFA-Net baseline)")
card(512, 964, "2.420", C_OUT, "NH-HAZE · NIQE (lower is better)",
     "best no-reference naturalness", "among all compared methods")
card(994, 1446, "0.752 M", C_DGC, "trainable parameters",
     "$\\approx$5 % overhead on FFA-Net backbone;", "DAV2 (24.18 M) stays frozen")
card(1476, 1928, "12.52 / 0.4903", C_ATT, "NH-HAZE · PSNR / SSIM",
     "2nd in PSNR (behind much larger DNMGDT);", "3rd in SSIM · lightest model compared")

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
out_pdf = r"E:\dsh work\fig1_work\fig1_overview_v1.pdf"
out_png = r"E:\dsh work\fig1_work\fig1_overview_v1.png"
fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved:", out_pdf)
print("saved:", out_png)
