# -*- coding: utf-8 -*-
"""
Figure 2 (corrected, v4) for JVCIR revision.
Measurement-driven layout: every box is sized from the rendered text extent,
so no label can overflow its frame.

Content fixes vs. the old fig2_preview.png:
  1) DG-CCA internals: QKV cross-modal channel attention
     (Q from RGB, K/V from depth; each f = 1x1 conv + 3x3 depthwise conv;
      L2 norm on Q,K -> CxC softmax attention matrix -> Wp projection + residual).
  2) A SINGLE DG-CCA module, after shallow feature extraction, before group blocks.
  3) Depth chain: Frozen DAV2 -> Depth Map (min-max) -> Soft Sky Suppression
     (D*clip(D/tau,0,1), tau=0.1) -> Depth Adapter -> K/V of DG-CCA.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

fig, ax = plt.subplots(figsize=(18.6, 7.0), dpi=110)
ax.set_xlim(0, 1860)
ax.set_ylim(0, 700)
ax.axis("off")
fig.patch.set_facecolor("white")
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = ax.transData.inverted()

# ---- palette ---------------------------------------------------------------
C_RGB,  F_RGB  = "#3B6EA5", "#E8F0FE"   # RGB stream (blue)
C_DEP,  F_DEP  = "#B26B2B", "#FBE5D6"   # depth stream (orange)
C_FRZ,  F_FRZ  = "#A34A45", "#FCE8E6"   # frozen module (red)
C_DGC,  F_DGC  = "#B45F06", "#FDF2E3"   # DG-CCA module (amber)
C_ATT,  F_ATT  = "#6A329F", "#F3E8FD"   # attention ops (purple)
C_OUT,  F_OUT  = "#2E7D32", "#E6F4EA"   # output (green)
INK  = "#1a1a2e"
GRAY = "#666666"

# ---- measurement helpers ---------------------------------------------------
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


def mbox(cx, cy, text, fc, ec, fs=8.2, bold=False, pad_x=30, pad_y=20,
         min_w=0, min_h=0, ls="-", lw=1.5, tc=INK, rsize=7):
    """Box auto-sized to fit its text."""
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


def arr(x1, y1, x2, y2, color="#555555", lw=1.6, ls="-", ms=14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=ms, linewidth=lw, color=color, linestyle=ls))


def seg(pts, color="#555555", lw=1.6, ls="-"):
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            color=color, lw=lw, linestyle=ls, solid_capstyle="round")


def circle_op(cx, cy, r, symbol, ec, fs=12):
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=ec, linewidth=1.6))
    ax.text(cx, cy - 0.5, symbol, ha="center", va="center",
            fontsize=fs, color=ec, family="DejaVu Sans")


# ---- real scene thumbnails (RESIDE-OTS paired images) ------------------------
import matplotlib.image as mpimg

HAZY_PATH = r"E:\Dehazing_Project\data\RESIDE\OTS\hazy\0025_0.8_0.2.jpg"
CLEAR_PATH = r"E:\Dehazing_Project\data\RESIDE\OTS\clear\0025.jpg"
_imgcache = {}


def load_crop(path, aspect):
    """Load image and center-crop to the requested width/height aspect."""
    if path in _imgcache:
        img = _imgcache[path]
    else:
        img = mpimg.imread(path)
        _imgcache[path] = img
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
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False,
                           edgecolor=frame, linewidth=1.6))

# ============================================================================
# (a) Overall architecture
# ============================================================================
ax.text(30, 668, "(a) Overall architecture — a single DG-CCA module after shallow feature extraction",
        fontsize=11, weight="bold", family="DejaVu Sans", color=INK)

RGB_CY = 470          # RGB stream centerline
DEP_CY = 195          # depth stream centerline

# --- input thumbnail ---
thumb(40, 422, 110, 95, HAZY_PATH)
ax.text(95, 546, "Hazy Image", fontsize=8.6, weight="bold", family="DejaVu Sans",
        color=INK, ha="center")
ax.text(95, 530, "$I\\in\\mathbb{R}^{3\\times H\\times W}$", fontsize=7.6,
        family="DejaVu Sans", color=GRAY, ha="center")

# --- measure RGB-stream boxes ---
shallow_txt = "Shallow Conv (3$\\times$3)\nshallow feature\nextraction"
ffa_txt = "FFA-Net Group Blocks\nmulti-scale feature\nextraction & reconstruction"
dgc_l1 = "DG-CCA  $\\times$1"
dgc_l2 = "Depth-Guided Cross-modal\nChannel Attention"
dgc_l3 = "$Q\\!\\leftarrow\\!F_{rgb}$;  $K,V\\!\\leftarrow\\!F_{d}$;  $A\\in\\mathbb{R}^{C\\times C}$"

w_sh = measure(shallow_txt, 8.2, True)[0] + 26
w_ff = measure(ffa_txt, 8.0, True)[0] + 26
w_dg = max(measure(dgc_l1, 12, True)[0],
           measure(dgc_l2, 8.4, False)[0],
           measure(dgc_l3, 7.8, False)[0]) + 30
w_chip = measure("$F_{rgb}$", 10.5, True)[0] + 30

# fit: start after input thumb, end before output thumb, within panel (a)
X_START, X_END = 182, 1196
TH_W = 110
total = w_sh + w_chip + w_dg + w_chip + w_ff
g = max(((X_END - TH_W - 26) - X_START - total) / 5.0, 14.0)

x = X_START
sh_cx = x + w_sh / 2; x += w_sh + g
chip1_cx = x + w_chip / 2; x += w_chip + g
dgc_cx = x + w_dg / 2; x += w_dg + g
chip2_cx = x + w_chip / 2; x += w_chip + g
ffa_cx = x + w_ff / 2; x += w_ff
out_x0 = x + g

# draw RGB boxes (aligned heights)
shallow = mbox(sh_cx, RGB_CY, shallow_txt, F_RGB, C_RGB, fs=8.2, bold=True, min_h=70)
chip1 = mbox(chip1_cx, RGB_CY, "$F_{rgb}$", F_RGB, C_RGB, fs=10.5, bold=True, min_w=w_chip, min_h=42)
ax.text(chip1_cx, chip1["y1"] + 11, "$C\\!\\times\\!H\\!\\times\\!W$,  $C=64$", fontsize=6.9,
        family="DejaVu Sans", color=C_RGB, ha="center")

# DG-CCA module box (single instance)
h_dg = 118
ax.add_patch(FancyBboxPatch((dgc_cx - w_dg / 2, RGB_CY - h_dg / 2), w_dg, h_dg,
             boxstyle="round,pad=0.02,rounding_size=8",
             linewidth=2.0, edgecolor=C_DGC, facecolor=F_DGC))
ax.text(dgc_cx, RGB_CY + 40, dgc_l1, fontsize=12, weight="bold",
        family="DejaVu Sans", color=C_DGC, ha="center")
ax.text(dgc_cx, RGB_CY + 10, dgc_l2, fontsize=8.4, family="DejaVu Sans",
        color="#7a4a06", ha="center", linespacing=1.3)
ax.text(dgc_cx, RGB_CY - 40, dgc_l3, fontsize=7.8, family="DejaVu Sans",
        color="#7a4a06", ha="center")
dgc = dict(cx=dgc_cx, cy=RGB_CY, w=w_dg, h=h_dg,
           x0=dgc_cx - w_dg / 2, x1=dgc_cx + w_dg / 2,
           y0=RGB_CY - h_dg / 2, y1=RGB_CY + h_dg / 2)
ax.add_patch(Rectangle((dgc["x0"] - 8, dgc["y0"] - 8), w_dg + 16, h_dg + 16,
                       fill=False, edgecolor=C_DGC, linewidth=1.2, linestyle=(0, (4, 3))))

chip2 = mbox(chip2_cx, RGB_CY, "$F_{out}$", F_OUT, C_OUT, fs=10.5, bold=True, min_w=w_chip, min_h=42)
ffa = mbox(ffa_cx, RGB_CY, ffa_txt, F_RGB, C_RGB, fs=8.0, bold=True, min_h=70)
ax.text(ffa_cx, ffa["y0"] - 12, "trainable", fontsize=7.6, family="DejaVu Sans",
        color=C_RGB, ha="center", style="italic")

# output thumbnail
thumb(out_x0, 422, TH_W, 95, CLEAR_PATH)
ax.text(out_x0 + TH_W / 2, 546, "Dehazed Output", fontsize=8.6, weight="bold",
        family="DejaVu Sans", color=INK, ha="center")
ax.text(out_x0 + TH_W / 2, 530, "$\\hat{I}$", fontsize=8.0, family="DejaVu Sans",
        color=C_OUT, ha="center")

# RGB arrows
arr(150, RGB_CY, shallow["x0"] - 2, RGB_CY, color=C_RGB)
arr(shallow["x1"] + 2, RGB_CY, chip1["x0"] - 2, RGB_CY, color=C_RGB)
arr(chip1["x1"] + 2, RGB_CY, dgc["x0"] - 10, RGB_CY, color=C_RGB)
arr(dgc["x1"] + 10, RGB_CY, chip2["x0"] - 2, RGB_CY, color=C_RGB)
arr(chip2["x1"] + 2, RGB_CY, ffa["x0"] - 2, RGB_CY, color=C_RGB)
arr(ffa["x1"] + 2, RGB_CY, out_x0 - 4, RGB_CY, color=C_RGB)

# --- Stage-2 note (top-left) ---
s2_txt = ("Stage 2 (last 20 epochs): regional decoupled\n"
          "fine-tuning — Targeted Sky TV Loss applied\n"
          "only to masked sky regions ($\\tau=0.1$)")
s2w = measure(s2_txt, 7.9, False)[0] + 30
s2h = measure(s2_txt, 7.9, False)[1] + 18
ax.add_patch(FancyBboxPatch((30, 560), s2w, s2h,
             boxstyle="round,pad=0.02,rounding_size=6",
             linewidth=1.1, edgecolor=GRAY, facecolor="#FAFAFA", linestyle=(0, (4, 3))))
ax.text(30 + s2w / 2, 560 + s2h / 2, s2_txt, fontsize=7.9, family="DejaVu Sans",
        color="#444444", ha="center", va="center", linespacing=1.35)

# --- depth stream ---
dav2_txt = "Frozen DAV2\nDepth Anything V2 (small)\n(no gradients)"
dmap_txt = "Depth Map\n$D\\in\\mathbb{R}^{1\\times H\\times W}$\nmin-max $\\rightarrow [0,1]$"
sky_txt = "Soft Sky Suppression\n$D\\cdot\\mathrm{clip}(D/\\tau,\\,0,\\,1)$\n$\\tau=0.1$, attenuates sky noise"
adp_txt = "Depth Adapter\n3$\\times$3 conv (C/2) + ReLU\n3$\\times$3 conv $\\rightarrow$ C"

w_dav2 = measure(dav2_txt, 8.2, True)[0] + 26
w_dmap = measure(dmap_txt, 8.2, True)[0] + 26
w_sky = measure(sky_txt, 8.0, True)[0] + 26
w_adp = measure(adp_txt, 8.0, True)[0] + 26

DX = 150
dg2 = 28
dav2_cx = DX + w_dav2 / 2
dmap_cx = DX + w_dav2 + dg2 + w_dmap / 2
sky_cx = DX + w_dav2 + w_dmap + 2 * dg2 + w_sky / 2
adp_cx = DX + w_dav2 + w_dmap + w_sky + 3 * dg2 + w_adp / 2

dav2 = mbox(dav2_cx, DEP_CY, dav2_txt, F_FRZ, C_FRZ, fs=8.2, bold=True, min_h=68)
dmap = mbox(dmap_cx, DEP_CY, dmap_txt, F_DEP, C_DEP, fs=8.2, bold=True, min_h=68)
sky = mbox(sky_cx, DEP_CY, sky_txt, F_DEP, C_DEP, fs=8.0, bold=True, min_h=68, ls=(0, (4, 3)))
adp = mbox(adp_cx, DEP_CY, adp_txt, F_DEP, C_DEP, fs=8.0, bold=True, min_h=68)

ax.text((dav2_cx + adp_cx) / 2, dav2["y0"] - 22,
        "depth prior branch — frozen encoder, trainable adapter",
        fontsize=8, family="DejaVu Sans", color="#8a5a2b", ha="center", style="italic")

# input branch down to DAV2
seg([(95, 422), (95, DEP_CY)], color=C_FRZ, lw=1.5)
arr(95, DEP_CY, dav2["x0"] - 2, DEP_CY, color=C_FRZ, lw=1.5)
ax.text(78, 320, "same input $I$", fontsize=7.2, family="DejaVu Sans", color=GRAY,
        ha="center", va="center", rotation=90)

# depth arrows
arr(dav2["x1"] + 2, DEP_CY, dmap["x0"] - 2, DEP_CY, color=C_DEP)
arr(dmap["x1"] + 2, DEP_CY, sky["x0"] - 2, DEP_CY, color=C_DEP)
arr(sky["x1"] + 2, DEP_CY, adp["x0"] - 2, DEP_CY, color=C_DEP)

# adapter -> up into DG-CCA (K/V inputs)
elb_y = DEP_CY + 92
seg([(adp_cx, adp["y1"] + 2), (adp_cx, elb_y), (dgc_cx, elb_y)], color=C_DEP, lw=1.7)
arr(dgc_cx, elb_y, dgc_cx, dgc["y0"] - 12, color=C_DEP, lw=1.7)
ax.text((adp_cx + dgc_cx) / 2, elb_y + 10,
        "$F_{d}\\in\\mathbb{R}^{C\\times H\\times W}$  $\\rightarrow$  K, V of DG-CCA",
        fontsize=8, family="DejaVu Sans", color=C_DEP, ha="center")

# --- zoom connector (arc through the empty band above the RGB stream) ---
ax.add_patch(FancyArrowPatch((dgc["x1"] + 8, dgc["y1"] + 8), (1250, 540),
             connectionstyle="arc3,rad=-0.16", arrowstyle="-|>",
             mutation_scale=13, linewidth=1.3, color=C_DGC, linestyle=(0, (5, 3))))
ax.text(940, 600, "detail view", fontsize=7.6, family="DejaVu Sans",
        color=C_DGC, ha="center", style="italic")

# --- legend ---
def swatch(x, y, fc, ec, ls="-"):
    ax.add_patch(Rectangle((x, y), 26, 16, facecolor=fc, edgecolor=ec,
                           linewidth=1.1, linestyle=ls))
swatch(30, 88, F_RGB, C_RGB);   ax.text(64, 96, "RGB stream (trainable)", fontsize=7.6, family="DejaVu Sans", va="center")
swatch(240, 88, F_DEP, C_DEP);  ax.text(274, 96, "depth features", fontsize=7.6, family="DejaVu Sans", va="center")
swatch(410, 88, F_FRZ, C_FRZ);  ax.text(444, 96, "frozen module", fontsize=7.6, family="DejaVu Sans", va="center")
swatch(580, 88, F_ATT, C_ATT);  ax.text(614, 96, "attention ops", fontsize=7.6, family="DejaVu Sans", va="center")
swatch(30, 54, F_DGC, C_DGC);   ax.text(64, 62, "DG-CCA module ($\\times$1)", fontsize=7.6, family="DejaVu Sans", va="center")
swatch(240, 54, F_OUT, C_OUT);  ax.text(274, 62, "output / restored", fontsize=7.6, family="DejaVu Sans", va="center")
seg([(415, 62), (455, 62)], color=C_DEP, lw=1.4, ls=(0, (4, 3)))
ax.text(462, 62, "soft sky suppression (preprocessing)", fontsize=7.6, family="DejaVu Sans", va="center")

# ============================================================================
# (b) DG-CCA internals — QKV cross-modal channel attention
# ============================================================================
PB_X0, PB_Y0, PB_W, PB_H = 1250, 40, 580, 620
ax.add_patch(FancyBboxPatch((PB_X0, PB_Y0), PB_W, PB_H,
             boxstyle="round,pad=0.02,rounding_size=9",
             linewidth=1.3, edgecolor=C_DGC, facecolor="#FDFBF7", linestyle=(0, (5, 3))))
ax.text(PB_X0 + PB_W / 2, PB_Y0 + PB_H - 22, "DG-CCA Module — internal structure",
        fontsize=10.5, weight="bold", family="DejaVu Sans", color=C_DGC, ha="center")

QX, KX, VX = 1350, 1540, 1730          # branch column centers
TOP = PB_Y0 + PB_H                       # 660

# input stubs
arr(PB_X0 - 22, TOP - 70, QX, TOP - 70, color=C_RGB, lw=1.6)
seg([(QX, TOP - 70), (QX, TOP - 78)], color=C_RGB, lw=1.6)
ax.text(PB_X0 + 8, TOP - 60, "$F_{rgb}\\in\\mathbb{R}^{C\\times H\\times W}$",
        fontsize=7.6, family="DejaVu Sans", color=C_RGB)
arr(PB_X0 - 22, TOP - 40, VX, TOP - 40, color=C_DEP, lw=1.6)
arr(KX, TOP - 40, KX, TOP - 78, color=C_DEP, lw=1.6)
arr(VX, TOP - 40, VX, TOP - 78, color=C_DEP, lw=1.6)
ax.text(PB_X0 + 8, TOP - 30, "$F_{d}\\in\\mathbb{R}^{C\\times H\\times W}$",
        fontsize=7.6, family="DejaVu Sans", color=C_DEP)

# branch boxes (auto-sized, uniform width per column set)
conv_txts = ["$f_q$: 1$\\times$1 conv", "$f_k$: 1$\\times$1 conv", "$f_v$: 1$\\times$1 conv"]
w_conv = max(measure(t, 8.4, False)[0] for t in conv_txts) + 28
w_dw = measure("3$\\times$3 DW conv", 8.4, False)[0] + 28
w_col = max(w_conv, w_dw, 108)
h_col = 36
gap_col = 16

y1 = TOP - 80                                   # conv1 top
c1 = [mbox(QX, y1 - h_col / 2, conv_txts[0], F_RGB, C_RGB, fs=8.4, min_w=w_col, min_h=h_col, pad_x=10),
      mbox(KX, y1 - h_col / 2, conv_txts[1], F_DEP, C_DEP, fs=8.4, min_w=w_col, min_h=h_col, pad_x=10),
      mbox(VX, y1 - h_col / 2, conv_txts[2], F_DEP, C_DEP, fs=8.4, min_w=w_col, min_h=h_col, pad_x=10)]
y2 = y1 - h_col - gap_col
c2 = [mbox(QX, y2 - h_col / 2, "3$\\times$3 DW conv", F_RGB, C_RGB, fs=8.4, min_w=w_col, min_h=h_col, pad_x=10),
      mbox(KX, y2 - h_col / 2, "3$\\times$3 DW conv", F_DEP, C_DEP, fs=8.4, min_w=w_col, min_h=h_col, pad_x=10),
      mbox(VX, y2 - h_col / 2, "3$\\times$3 DW conv", F_DEP, C_DEP, fs=8.4, min_w=w_col, min_h=h_col, pad_x=10)]
y3 = y2 - h_col - gap_col
l2q = mbox(QX, y3 - h_col / 2, "L2 norm", F_RGB, C_RGB, fs=8.4, min_w=w_col, min_h=h_col, pad_x=10)
l2k = mbox(KX, y3 - h_col / 2, "L2 norm", F_DEP, C_DEP, fs=8.4, min_w=w_col, min_h=h_col, pad_x=10)
y4 = y3 - h_col - gap_col
w_chipb = 68
qhat = mbox(QX, y4 - 17, "$\\hat{Q}$", F_RGB, C_RGB, fs=11, bold=True, min_w=w_chipb, min_h=34)
khat = mbox(KX, y4 - 17, "$\\hat{K}$", F_DEP, C_DEP, fs=11, bold=True, min_w=w_chipb, min_h=34)
vchip = mbox(VX, y3 - h_col / 2, "$V$", F_DEP, C_DEP, fs=11, bold=True, min_w=w_chipb, min_h=34)

for cx_, col_ in ((QX, C_RGB), (KX, C_DEP), (VX, C_DEP)):
    arr(cx_, y1 - h_col - 1, cx_, y2 + 1, color=col_, ms=11)
    arr(cx_, y2 - h_col - 1, cx_, y3 + 1, color=col_, ms=11)
arr(QX, y3 - h_col - 1, QX, qhat["y1"] + 1, color=C_RGB, ms=11)
arr(KX, y3 - h_col - 1, KX, khat["y1"] + 1, color=C_DEP, ms=11)

# --- attention computation ---
MY = 250                       # matmul row
MX = (QX + KX) / 2             # matmul circle centered between Q and K drops
circle_op(MX, MY, 14, "$\\times$", C_ATT, fs=11)
seg([(QX, qhat["y0"] - 2), (QX, MY)], color=C_ATT, lw=1.6)
arr(QX, MY, MX - 16, MY, color=C_ATT, lw=1.6)
seg([(KX, khat["y0"] - 2), (KX, MY)], color=C_ATT, lw=1.6)
arr(KX, MY, MX + 16, MY, color=C_ATT, lw=1.6)
ax.text(MX, MY + 36, "$\\alpha\\cdot\\hat{Q}\\hat{K}^{\\top}$",
        fontsize=8.4, family="DejaVu Sans", color=C_ATT, ha="center")
ax.text(MX, MY + 58, "($\\alpha$: learnable temperature)",
        fontsize=7.0, family="DejaVu Sans", color=C_ATT, ha="center")

# softmax below MX
sm_txt = "softmax\n(FP32)"
smw = measure(sm_txt, 8.4, True)[0] + 26
smh = measure(sm_txt, 8.4, True)[1] + 14
sm = mbox(MX, MY - 66, sm_txt, F_ATT, C_ATT, fs=8.4, bold=True, pad_x=26, pad_y=14)
arr(MX, MY - 14, MX, sm["y1"] + 2, color=C_ATT, lw=1.6)

# attention matrix A (grid icon) right of softmax
gy = sm["cy"]
gx0 = sm["x1"] + 30
arr(sm["x1"] + 2, gy, gx0 - 2, gy, color=C_ATT, lw=1.6)
GS = 38
ax.add_patch(Rectangle((gx0, gy - GS / 2), GS, GS, facecolor="white", edgecolor=C_ATT, linewidth=1.3))
for i in range(1, 4):
    seg([(gx0 + i * GS / 4, gy - GS / 2), (gx0 + i * GS / 4, gy + GS / 2)], color=C_ATT, lw=0.7)
    seg([(gx0, gy - GS / 2 + i * GS / 4), (gx0 + GS, gy - GS / 2 + i * GS / 4)], color=C_ATT, lw=0.7)
for (ccx, ccy) in [(0, 3), (2, 1), (3, 2)]:
    ax.add_patch(Rectangle((gx0 + ccx * GS / 4, gy - GS / 2 + ccy * GS / 4),
                           GS / 4, GS / 4, facecolor="#D9C4F0", edgecolor="none"))
ax.text(gx0 + GS / 2, gy + GS / 2 + 12, "$A\\in\\mathbb{R}^{C\\times C}$", fontsize=8,
        family="DejaVu Sans", color=C_ATT, ha="center")

# matmul with V
M2X = gx0 + GS + 42
arr(gx0 + GS + 2, gy, M2X - 15, gy, color=C_ATT, lw=1.6)
circle_op(M2X, gy, 13, "$\\times$", C_ATT, fs=11)
seg([(VX, vchip["y0"] - 2), (VX, gy)], color=C_ATT, lw=1.6)
arr(VX, gy, M2X + 15, gy, color=C_ATT, lw=1.6)
ax.text(M2X, gy + 28, "$A\\,V$", fontsize=7.8, family="DejaVu Sans", color=C_ATT, ha="center")

# W_p -> plus -> F_out (row below)
WY = 117
wp_txt = "$W_{p}$  (1$\\times$1 conv)"
wpw = measure(wp_txt, 8.4, True)[0] + 26
# place W_p so its right edge sits directly under the A·V matmul circle
wp_cx = M2X - (measure(wp_txt, 8.4, True)[0] + 26) / 2
wp = mbox(wp_cx, WY, wp_txt, F_ATT, C_ATT, fs=8.4, bold=True, pad_x=26, pad_y=14)
arr(M2X, gy - 15, M2X, wp["y1"] + 2, color=C_ATT, lw=1.6)

PLUS = wp["x0"] - 40
arr(wp["x0"] - 2, WY, PLUS + 15, WY, color=C_ATT, lw=1.6)
circle_op(PLUS, WY, 13, "$+$", C_OUT, fs=12)

# residual from F_rgb input
RX = PB_X0 + 14
seg([(RX, TOP - 70), (RX, WY)], color=C_OUT, lw=1.5, ls=(0, (4, 3)))
arr(RX, WY, PLUS - 15, WY, color=C_OUT, lw=1.5, ls=(0, (4, 3)))
ax.text(RX + 9, 360, "residual $+F_{rgb}$", fontsize=7.4, family="DejaVu Sans",
        color=C_OUT, rotation=90, va="center")

# F_out chip below the plus
fout = mbox(PLUS, WY - 52, "$F_{out}$", F_OUT, C_OUT, fs=10.5, bold=True, min_w=74, min_h=32)
arr(PLUS, WY - 13, PLUS, fout["y1"] + 2, color=C_OUT, lw=1.6)
ax.text(fout["x1"] + 16, fout["cy"],
        "$F_{out}=W_{p}(A\\,V)+F_{rgb}$   $\\rightarrow$  FFA-Net group blocks",
        fontsize=8.6, family="DejaVu Sans", color=INK, va="center")

# footer notes under panel (b)
ax.text(PB_X0 + PB_W / 2, 26,
        "$Q=f_q(F_{rgb})$;  $K=f_k(F_d)$;  $V=f_v(F_d)$ — each $f$: 1$\\times$1 conv $\\rightarrow$ 3$\\times$3 depthwise conv (no bias)",
        fontsize=7.9, family="DejaVu Sans", color=GRAY, ha="center")
ax.text(PB_X0 + PB_W / 2, 9,
        "L2 norm along spatial dim on $Q,K$;   softmax in FP32;   C = 64, 18,113 params",
        fontsize=7.9, family="DejaVu Sans", color=GRAY, ha="center")

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
out_pdf = r"E:\dsh work\fig2_work\fig2_architecture_v2.pdf"
out_png = r"E:\dsh work\fig2_work\fig2_preview_v2.png"
fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved:", out_pdf)
print("saved:", out_png)
