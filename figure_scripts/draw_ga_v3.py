# -*- coding: utf-8 -*-
"""
Graphical Abstract v3 (minimal Elsevier style, 3:1 banner, 1500x500).
Photos smaller; center enriched: Q / K,V stream labels, param line inside
DG-CCA, Depth label, SkyTV tag with dashed connector, stacked big numbers.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

fig, ax = plt.subplots(figsize=(15, 5), dpi=110)
ax.set_xlim(0, 1500)
ax.set_ylim(0, 500)
ax.axis("off")
fig.patch.set_facecolor("white")
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = ax.transData.inverted()

C_RGB, F_RGB = "#3B6EA5", "#E8F0FE"
C_DEP, F_DEP = "#B26B2B", "#FBE5D6"
C_ATT, F_ATT = "#6A329F", "#F3E8FD"
C_DGC        = "#B45F06"
C_SKY        = "#A34A45"
C_OUT        = "#2E7D32"
INK  = "#1a1a2e"
GRAY = "#555555"

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

def arr(x1, y1, x2, y2, color, lw=2.4, ls="-", ms=22):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=ms, linewidth=lw, color=color, linestyle=ls))

def seg(pts, color, lw=2.2, ls="-"):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
            lw=lw, linestyle=ls, solid_capstyle="round")

# ---- photos (smaller now: 190x230) -------------------------------------------
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

PW_, PH_ = 190, 230
def thumb(x0, y0, path, edge):
    ax.imshow(load_crop(path, PW_ / PH_), extent=(x0, x0 + PW_, y0, y0 + PH_),
              interpolation="bilinear")
    ax.add_patch(Rectangle((x0, y0), PW_, PH_, fill=False, edgecolor=edge, linewidth=2.0))

thumb(45, 200, HAZY_PATH, "#9A9A9A")
thumb(1265, 200, CLEAR_PATH, C_OUT)
ax.text(45 + PW_ / 2, 172, "Hazy", fontsize=15, weight="bold",
        family="DejaVu Sans", color=GRAY, ha="center")
ax.text(1265 + PW_ / 2, 172, "Dehazed", fontsize=15, weight="bold",
        family="DejaVu Sans", color=C_OUT, ha="center")

# ---- stacked result numbers under the dehazed photo -----------------------------
ax.text(1265 + PW_ / 2, 130, "+2.35 dB", fontsize=25, weight="bold",
        family="DejaVu Sans", color=C_RGB, ha="center")
ax.text(1265 + PW_ / 2, 90, "0.75 M", fontsize=23, weight="bold",
        family="DejaVu Sans", color=C_DGC, ha="center")

# ---- center: the single DG-CCA module ------------------------------------------
BX0, BY0, BW, BH = 688, 260, 264, 140
ax.add_patch(FancyBboxPatch((BX0, BY0), BW, BH,
             boxstyle="round,pad=0.02,rounding_size=10",
             linewidth=2.6, edgecolor=C_ATT, facecolor=F_ATT))
ax.text(BX0 + BW / 2, BY0 + 112, "DG-CCA $\\times$1", fontsize=21, weight="bold",
        family="DejaVu Sans", color=C_ATT, ha="center")
ax.text(BX0 + BW / 2, BY0 + 68,
        "$Q$ ← RGB   ·   $K$,$V$ ← depth   →   $A$ ($C\\!\\times\\!C$)",
        fontsize=9.2, family="DejaVu Sans", color="#4a2a75", ha="center")
ax.text(BX0 + BW / 2, BY0 + 32, "18,113 params · $\\approx$5% of backbone",
        fontsize=8.5, family="DejaVu Sans", color="#6a4a95", ha="center",
        style="italic")

# ---- Frozen DAV2 (depth stream) ------------------------------------------------
DW, DH = 185, 56
DX0, DY0 = 360, 132
ax.add_patch(FancyBboxPatch((DX0, DY0), DW, DH,
             boxstyle="round,pad=0.02,rounding_size=8",
             linewidth=1.8, edgecolor="#777777", facecolor="#EFEFEF"))
ax.text(DX0 + DW / 2, DY0 + DH / 2, "Frozen DAV2", fontsize=13, weight="bold",
        family="DejaVu Sans", color="#444444", ha="center", va="center")

# ---- arrows ----------------------------------------------------------------------
# RGB path (Q stream): hazy -> DG-CCA
arr(237, 365, BX0 - 4, 345, color=C_RGB, lw=2.8)
ax.text(648, 323, "$Q$", fontsize=13, weight="bold", family="DejaVu Sans",
        color=C_RGB, ha="center")
# depth path: hazy -> DAV2 -> K,V into DG-CCA
seg([(237, 270), (320, 270), (320, 160)], color=C_DEP, lw=2.4)
arr(320, 160, DX0 - 4, 160, color=C_DEP, lw=2.4)
seg([(DX0 + DW + 2, 160), (760, 160)], color=C_DEP, lw=2.4)
arr(760, 164, 760, BY0 - 4, color=C_DEP, lw=2.8)
ax.text(648, 184, "Depth", fontsize=11, weight="bold", family="DejaVu Sans",
        color=C_DEP, ha="center")
ax.text(786, 208, "$K$,$V$", fontsize=11.5, weight="bold", family="DejaVu Sans",
        color=C_DEP)
# output path: DG-CCA -> dehazed
arr(BX0 + BW + 4, 330, 1262, 330, color=C_RGB, lw=2.8)

# ---- SkyTV tag --------------------------------------------------------------------
sw, _ = measure("SkyTV ($\\tau$=0.1)", 11, True)
SW_, SH_ = sw + 26, 38
SX, SY = 1030 - SW_ / 2, 81
ax.add_patch(FancyBboxPatch((SX, SY), SW_, SH_,
             boxstyle="round,pad=0.02,rounding_size=9",
             linewidth=1.6, edgecolor=C_SKY, facecolor="white", linestyle=(0, (4, 3))))
ax.text(1030, SY + SH_ / 2, "SkyTV ($\\tau$=0.1)", fontsize=11, weight="bold",
        family="DejaVu Sans", color=C_SKY, ha="center", va="center")
arr(1000, SY + SH_ + 2, 878, BY0 + 2, color=C_SKY, lw=1.3, ls=(0, (4, 3)), ms=12)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
OUT = r"E:\dsh work\fig1_work"
fig.savefig(OUT + r"\ga_v3_master.png", format="png", dpi=300, facecolor="white")
fig.savefig(OUT + r"\fig1_overview_v3.pdf", format="pdf", facecolor="white")
print("saved master png + pdf")

from PIL import Image
im = Image.open(OUT + r"\ga_v3_master.png")
print("master size:", im.size)
im.resize((1500, 500), Image.LANCZOS).save(
    OUT + r"\fig1_overview_v3.png", dpi=(300, 300))
print("saved:", OUT + r"\fig1_overview_v3.png (1500x500 @300dpi)")
