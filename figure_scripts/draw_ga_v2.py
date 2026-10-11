# -*- coding: utf-8 -*-
"""
Graphical Abstract (v2, minimal Elsevier style) — 3 blocks, 3:1 banner.
Canvas = 1500x500 data units == 1500x500 px at final size.
Left: hazy input photo. Center: two streams -> ONE DG-CCA module (+SkyTV tag).
Right: dehazed photo + two big numbers (+2.35 dB, 0.75 M).
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

# ---- photos (RESIDE-OTS 0025 pair) -------------------------------------------
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

def thumb(x0, y0, w, h, path, edge):
    ax.imshow(load_crop(path, w / h), extent=(x0, x0 + w, y0, y0 + h),
              interpolation="bilinear")
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=edge, linewidth=2.0))

TY0, TH = 120, 340                      # photo band
thumb(40, TY0, 300, TH, HAZY_PATH, "#9A9A9A")
thumb(1160, TY0, 300, TH, CLEAR_PATH, "#2E7D32")
ax.text(190, 92, "Hazy", fontsize=16, weight="bold", family="DejaVu Sans",
        color=GRAY, ha="center")
ax.text(1310, 92, "Dehazed", fontsize=16, weight="bold", family="DejaVu Sans",
        color="#2E7D32", ha="center")

# ---- center: the single DG-CCA module ----------------------------------------
BX0, BY0, BW, BH = 620, 190, 260, 130
ax.add_patch(FancyBboxPatch((BX0, BY0), BW, BH,
             boxstyle="round,pad=0.02,rounding_size=10",
             linewidth=2.6, edgecolor=C_ATT, facecolor=F_ATT))
ax.text(BX0 + BW / 2, BY0 + 92, "DG-CCA", fontsize=23, weight="bold",
        family="DejaVu Sans", color=C_ATT, ha="center")
ax.text(BX0 + BW / 2, BY0 + 38,
        "$Q$ ← RGB   ·   $K$,$V$ ← depth   →   $A$ ($C\\!\\times\\!C$)",
        fontsize=9.5, family="DejaVu Sans", color="#4a2a75", ha="center")

# ---- Frozen DAV2 (depth stream) ----------------------------------------------
DW, DH = 190, 54
DX0, DY0 = 655, 58
ax.add_patch(FancyBboxPatch((DX0, DY0), DW, DH,
             boxstyle="round,pad=0.02,rounding_size=8",
             linewidth=1.8, edgecolor="#777777", facecolor="#EFEFEF"))
ax.text(DX0 + DW / 2, DY0 + DH / 2, "Frozen DAV2", fontsize=13, weight="bold",
        family="DejaVu Sans", color="#444444", ha="center", va="center")

# ---- arrows -------------------------------------------------------------------
# RGB path: hazy -> DG-CCA
arr(343, 360, BX0 - 5, 280, color=C_RGB, lw=2.8)
# depth path: hazy -> DAV2 -> DG-CCA
seg([(343, 240), (480, 240), (480, DY0 + DH / 2)], color=C_DEP, lw=2.4)
arr(480, DY0 + DH / 2, DX0 - 4, DY0 + DH / 2, color=C_DEP, lw=2.4)
arr(DX0 + DW / 2, DY0 + DH + 3, DX0 + DW / 2, BY0 - 4, color=C_DEP, lw=2.8)
ax.text(DX0 + DW / 2 + 24, (DY0 + DH + BY0) / 2, "Depth", fontsize=11,
        weight="bold", family="DejaVu Sans", color=C_DEP, va="center")
# output path: DG-CCA -> dehazed
arr(BX0 + BW + 5, 255, 1156, 255, color=C_RGB, lw=2.8)

# ---- SkyTV tag (small, bottom-right of the module) ----------------------------
sw, sh_ = measure("SkyTV ($\\tau$=0.1)", 11, True)
PW, PH = sw + 26, 38
PX, PY = 985 - PW / 2, 352
ax.add_patch(FancyBboxPatch((PX, PY), PW, PH,
             boxstyle="round,pad=0.02,rounding_size=9",
             linewidth=1.6, edgecolor=C_SKY, facecolor="white", linestyle=(0, (4, 3))))
ax.text(985, PY + PH / 2, "SkyTV ($\\tau$=0.1)", fontsize=11, weight="bold",
        family="DejaVu Sans", color=C_SKY, ha="center", va="center")
arr(PX + 6, PY + PH - 3, BX0 + BW - 6, BY0 + 4, color=C_SKY, lw=1.3, ls=(0, (4, 3)), ms=12)

# ---- result numbers ------------------------------------------------------------
n1, n2 = "+2.35 dB", "0.75 M"
w1 = measure(n1, 25, True)[0]
w2 = measure(n2, 25, True)[0]
gap = 26
tot = w1 + gap + w2
cx = 1310
ax.text(cx - tot / 2 + w1 / 2, 42, n1, fontsize=25, weight="bold",
        family="DejaVu Sans", color=C_RGB, ha="center")
ax.text(cx + tot / 2 - w2 / 2, 42, n2, fontsize=25, weight="bold",
        family="DejaVu Sans", color=C_DGC, ha="center")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
OUT = r"E:\dsh work\fig1_work"
fig.savefig(OUT + r"\ga_v2_master.png", format="png", dpi=300, facecolor="white")
fig.savefig(OUT + r"\fig1_overview_v2.pdf", format="pdf", facecolor="white")
print("saved master png + pdf")

# exact 1500x500 GA version (downsampled from the 300-dpi master)
from PIL import Image
im = Image.open(OUT + r"\ga_v2_master.png")
print("master size:", im.size)
im.resize((1500, 500), Image.LANCZOS).save(
    OUT + r"\fig1_overview_v2.png", dpi=(300, 300))
print("saved:", OUT + r"\fig1_overview_v2.png (1500x500 @300dpi)")
