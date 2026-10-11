"""方案 A：经典双流水平展开式 (The Classic Dual-Stream)"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math, os

W, H = 2200, 1500
img = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(img)

try:
    font_title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 26)
    font_hdr   = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 18)
    font_norm  = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
except:
    font_title = font_hdr = font_norm = font_small = ImageFont.load_default()

def rbox(x1, y1, x2, y2, fill, outline=(0,0,0), width=2, radius=8):
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=width)

def txt(x, y, s, font=font_norm, fill=(0,0,0), anchor="mm"):
    b = draw.textbbox((0,0), s, font=font)
    tw, th = b[2]-b[0], b[3]-b[1]
    if anchor == "mm": draw.text((x-tw//2, y-th//2), s, fill=fill, font=font)
    elif anchor == "lm": draw.text((x, y-th//2), s, fill=fill, font=font)

def arr(x1,y1,x2,y2,color=(80,80,80),width=2):
    """Horizontal or vertical arrow with 90° path (L-shape if needed)."""
    if x1==x2 or y1==y2:
        draw.line([(x1,y1),(x2,y2)], fill=color, width=width)
        angle = math.atan2(y2-y1, x2-x1)
    else:
        # L-shaped: horizontal then vertical (or vice versa)
        if abs(x2-x1) > abs(y2-y1):
            mx, my = x2, y1
        else:
            mx, my = x1, y2
        draw.line([(x1,y1),(mx,my),(x2,y2)], fill=color, width=width)
        angle = math.atan2(y2-my, x2-mx)
    L=12
    ax = x2 - L*math.cos(angle-0.4)
    ay = y2 - L*math.sin(angle-0.4)
    bx = x2 - L*math.cos(angle+0.4)
    by = y2 - L*math.sin(angle+0.4)
    draw.polygon([(x2,y2),(ax,ay),(bx,by)], fill=color)

def harr(x1,x2,y,color=(80,80,80),width=2):
    draw.line([(x1,y),(x2,y)], fill=color, width=width)
    L=12
    draw.polygon([(x2,y),(x2-L,y-6),(x2-L,y+6)], fill=color)

def varr(x,y1,y2,color=(80,80,80),width=2):
    draw.line([(x,y1),(x,y2)], fill=color, width=width)
    L=12
    draw.polygon([(x,y2),(x-6,y2-L),(x+6,y2-L)], fill=color)

# ===== TITLE =====
txt(W//2, 35, "Architecture Overview — Classic Dual-Stream Design", font_title, (0,0,0))

# ========================
# UPPER TRACK: RGB Flow (水平)
# ========================
cy = 200
# Hazy Input
rbox(60, cy-35, 240, cy+35, (220,235,255), (60,100,180))
txt(150, cy, "Hazy I", font_hdr, (20,60,140))

harr(240, 320, cy)
# Group Process 1
rbox(320, cy-45, 560, cy+45, (240,248,255), (70,130,180))
txt(440, cy, "Group 1", font_hdr, (30,80,140))

harr(560, 640, cy)
# DG-CCA 1 (红色醒目)
rbox(640, cy-50, 840, cy+50, (255,225,235), (200,50,100), width=3)
txt(740, cy, "DG-CCA", font_hdr, (180,20,80))

harr(840, 920, cy)
# Group Process 2
rbox(920, cy-45, 1160, cy+45, (240,248,255), (70,130,180))
txt(1040, cy, "Group 2", font_hdr, (30,80,140))

harr(1160, 1240, cy)
# DG-CCA 2
rbox(1240, cy-50, 1440, cy+50, (255,225,235), (200,50,100), width=3)
txt(1340, cy, "DG-CCA", font_hdr, (180,20,80))

harr(1440, 1520, cy)
# Group Process 3
rbox(1520, cy-45, 1760, cy+45, (240,248,255), (70,130,180))
txt(1640, cy, "Group 3", font_hdr, (30,80,140))

harr(1760, 1840, cy)
# DG-CCA 3
rbox(1840, cy-50, 2040, cy+50, (255,225,235), (200,50,100), width=3)
txt(1940, cy, "DG-CCA", font_hdr, (180,20,80))

harr(2040, 2100, cy)
# Decoder
rbox(2100, cy-40, 2180, cy+40, (240,248,255), (70,130,180))
txt(2140, cy, "Decoder", font_hdr, (30,80,140))

# ========================
# OUTPUT
# ========================
# Arrow right + output box
harr(2180, 2260, cy)
# Restored Output
rbox(2260, cy-35, 2440, cy+35, (255,235,220), (180,80,40))
txt(2350, cy, "Restored Î", font_hdr, (140,50,20))

# ========================
# LOWER TRACK: Depth Flow
# ========================
dy = 430
# DAV2 box (全宽底部)
rbox(60, dy-30, 440, dy+30, (220,255,220), (60,130,60))
txt(250, dy, "Frozen DAV2 Depth Estimator", font_hdr, (20,100,20))

# Vertical connection: Hazy I down to DAV2
varr(150, cy+35, dy-30, (80,80,80), 1)

# Depth baseline (水平贯穿底部)
draw.line([(440, dy), (2100, dy)], fill=(80,140,80), width=3)
# label
txt(800, dy+25, "Depth Feature Baseline (贯穿 3 个 Group)", font_small, (60,120,60))

# Vertical injection arrows from depth baseline up to each DG-CCA
inject_x = [740, 1340, 1940]
for ix in inject_x:
    draw.line([(ix, dy), (ix, cy+50)], fill=(100,160,100), width=2)
    # arrowhead at top
    draw.polygon([(ix, cy+50), (ix-6, cy+50-12), (ix+6, cy+50-12)], fill=(100,160,100))

# ========================
# BOTTOM: DG-CCA Detail Callout
# ========================
bx, by = 100, 580
rbox(bx, by, bx+1000, by+400, (253,253,250), (120,120,120), width=1)
txt(bx+500, by+15, "DG-CCA Detail (per module)", font_hdr, (80,40,80))

# F_rgb box
rbox(bx+80, by+50, bx+220, by+85, (220,240,255), (60,100,160))
txt(bx+150, by+67, "F_rgb (C×H×W)", font_small, (50,80,140))

# F_d box
rbox(bx+780, by+50, bx+920, by+85, (220,255,220), (60,140,60))
txt(bx+850, by+67, "F_d (C×H×W)", font_small, (50,120,50))

# GAP arrows
harr(bx+220, bx+280, by+67, (80,80,80))
arr(bx+300, by+67, bx+300, by+120, (80,80,80))
harr(bx+770, bx+710, by+67, (80,80,80))
arr(bx+710, by+67, bx+710, by+120, (80,80,80))

# GAP boxes
rbox(bx+250, by+120, bx+370, by+155, (230,240,255), (60,100,160))
txt(bx+310, by+137, "GAP → z_rgb", font_small, (50,80,140))
rbox(bx+630, by+120, bx+750, by+155, (230,255,230), (60,140,60))
txt(bx+690, by+137, "GAP → z_d", font_small, (50,120,50))

# Concat arrows
harr(bx+370, bx+440, by+137, (80,80,80))
harr(bx+630, bx+560, by+137, (80,80,80))

# Concat box
rbox(bx+430, by+175, bx+570, by+210, (255,235,245), (160,40,90))
txt(bx+500, by+192, "Concat [z_rgb; z_d]", font_small, (150,30,80))

# MLP arrow
varr(bx+500, by+210, by+245)

# MLP box
rbox(bx+410, by+245, bx+590, by+280, (255,225,235), (160,30,80))
txt(bx+500, by+262, "FC→ReLU→FC→σ → A", font_small, (150,20,70))

# Multiply arrow
harr(bx+590, bx+650, by+262, (160,30,80))

# Out text
txt(bx+750, by+262, "F_out = A ⊙ F_rgb", font_norm, (140,20,70), "lm")

# Params note
txt(bx+500, by+320, "C=64,  ~8.5K params/module  |  3 modules × 8.5K = 25.5K total", font_small, (100,100,100))

# Arrow from detail back to main — connect MLP output to "channel-wise scaling"
draw.line([(bx+650, by+262), (bx+650, by+380), (bx+500, by+380)], fill=(160,30,80), width=1)
dash_len = 6
for x in range(bx+500, bx+800, dash_len*2):
    draw.line([(x, by+380), (min(x+dash_len, bx+800), by+380)], fill=(160,30,80), width=1)

# ========================
# SKY TV LOSS callout (right side)
# ========================
sx, sy = 1200, 580
rbox(sx, sy, sx+1020, sy+400, (255,255,238), (130,130,80), width=1)

txt(sx+510, sy+20, "Targeted Sky TV Fine-Tuning (Epoch 101–120)", font_hdr, (120,100,30))

# Mask box
rbox(sx+80, sy+70, sx+280, sy+110, (255,245,220), (160,100,40))
txt(sx+180, sy+90, "Depth Mask M (τ=0.1)", font_small, (140,80,30))

# Sky region示意
rbox(sx+80, sy+130, sx+280, sy+200, (235,245,255), (80,120,180))
txt(sx+180, sy+150, "Sky Region", font_small, (60,80,140))
txt(sx+180, sy+170, "(D ≥ τ → mask=1)", font_small, (80,100,160))

# TV Loss box
rbox(sx+420, sy+70, sx+720, sy+110, (255,240,225), (160,80,40))
txt(sx+570, sy+90, "Spatial TV Loss", font_small, (140,60,30))

# Formula
txt(sx+510, sy+155, "L_TV = Σ (1−M) · ||∇Î||₁", font_norm, (120,80,30))

# Total loss
txt(sx+510, sy+210, "L_rec = L1 + 0.05·L_edge", font_small, (80,80,80))
txt(sx+510, sy+240, "L_total = L_rec + λ · L_TV  (λ=0.01)", font_hdr, (140,30,30))

# Arrow linking detail to main loss
# Draw small adapter: from output to loss
arr(2140, cy, 2300, by+100, (80,80,80), 1)

# ===== SAVE =====
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper", "CVPR_2026_Submission_Template", "figs", "architecture_A_dual_stream.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
img.save(OUT)
print(f"Saved: {OUT}")