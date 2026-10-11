"""方案 C："T"型全局先验注入式 (The T-Shape Prior Injection)"""
from PIL import Image, ImageDraw, ImageFont
import math, os

W, H = 2000, 1500
img = Image.new("RGB", (W, H), (255,255,255))
d = ImageDraw.Draw(img)

try:
    ft = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 26)
    fh = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 18)
    fn = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    fs = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
except:
    ft=fh=fn=fs=ImageFont.load_default()

def rbox(x1,y1,x2,y2,fill,outline=(0,0,0),width=2,radius=8):
    d.rounded_rectangle([x1,y1,x2,y2], radius=radius, fill=fill, outline=outline, width=width)

def txt(x,y,s,font=fn,fill=(0,0,0),anchor="mm"):
    b=d.textbbox((0,0),s,font=font);tw,th=b[2]-b[0],b[3]-b[1]
    if anchor=="mm": d.text((x-tw//2,y-th//2),s,fill=fill,font=font)
    elif anchor=="lm": d.text((x,y-th//2),s,fill=fill,font=font)

def harr(x1,x2,y,c=(80,80,80),w=2):
    d.line([(x1,y),(x2,y)],fill=c,width=w)
    L=12; d.polygon([(x2,y),(x2-L,y-6),(x2-L,y+6)],fill=c)

def varr(x,y1,y2,c=(80,80,80),w=2):
    d.line([(x,y1),(x,y2)],fill=c,width=w)
    L=12; d.polygon([(x,y2),(x-6,y2-L),(x+6,y2-L)],fill=c)

# ===== TITLE =====
txt(W//2, 30, "Architecture Overview — T-Shape Prior Injection", ft, (0,0,0))

# ========================
# TOP CENTER: Hazy Input + DAV2 (上帝视角)
# ========================
# Big DAV2 box
rbox(400, 55, 1600, 130, (220,255,220), (60,130,60), width=3)
txt(1000, 75, "Frozen DAV2 — Foundation Depth Estimator (24.8M params)", fh, (20,100,20))
txt(1000, 110, "Global Geometry Prior (God's-Eye View)", fs, (40,120,40))

# Hazy Input above DAV2
rbox(750, 10, 1250, 48, (220,235,255), (60,100,180))
txt(1000, 29, "Hazy Input I (3×H×W)", fh, (20,60,140))
# Arrow down to DAV2
varr(1000, 48, 55, (80,80,80))

# ========================
# CENTER: Depth Map
# ========================
rbox(820, 155, 1180, 210, (235,255,235), (80,140,80))
txt(1000, 182, "Depth Map D (1×H×W)", fh, (30,100,30))

# Depth normalization + adapter
rbox(790, 235, 1210, 285, (230,250,230), (80,140,80))
txt(1000, 260, "Depth Feature: Normalize + 1×1 Conv Adapter → 64ch", fn, (40,100,40))

# ========================
# THREE VERTICAL ARROWS DOWN
# ========================
varr(700, 285, 380, (80,140,80), 2)
varr(1000, 285, 380, (80,140,80), 2)
varr(1300, 285, 380, (80,140,80), 2)

# Labels on arrows
txt(700, 330, "skip", fs, (80,140,80))
txt(1000, 330, "depth", fs, (80,140,80))
txt(1300, 330, "guide", fs, (80,140,80))

# ========================
# BOTTOM: RGB Processing Pipeline (水平)
# ========================
by = 400
# Group 1 + DG-CCA 1
rbox(100, by, 480, by+80, (240,248,255), (70,130,180))
txt(290, by+25, "Group Process 1", fh, (30,80,140))
txt(290, by+55, "RGB features", fs, (50,90,150))

rbox(530, by-5, 830, by+85, (255,225,235), (200,50,100), width=3)
txt(680, by+20, "DG-CCA 1", fh, (180,20,80))
txt(680, by+55, "F_rgb + F_d → GAP→MLP→σ→⊙", fs, (140,30,70))

# Arrow Group1 → DG-CCA1
harr(480, 530, by+40, (100,100,100))
# Depth injection into DG-CCA1
varr(700, 285, by-5, (80,140,80), 2)

# Group 2 + DG-CCA 2
rbox(900, by, 1280, by+80, (240,248,255), (70,130,180))
txt(1090, by+25, "Group Process 2", fh, (30,80,140))

rbox(1330, by-5, 1630, by+85, (255,225,235), (200,50,100), width=3)
txt(1480, by+20, "DG-CCA 2", fh, (180,20,80))

# Arrow DG-CCA1 → Group2
harr(830, 900, by+40, (100,100,100))
# Arrow Group2 → DG-CCA2
harr(1280, 1330, by+40, (100,100,100))
# Depth injection into DG-CCA2
varr(1000, 285, by-5, (80,140,80), 2)

# Group 3 + DG-CCA 3
rbox(100, by+150, 480, by+230, (240,248,255), (70,130,180))
txt(290, by+175, "Group Process 3", fh, (30,80,140))

rbox(530, by+145, 830, by+235, (255,225,235), (200,50,100), width=3)
txt(680, by+170, "DG-CCA 3", fh, (180,20,80))

# Arrow DG-CCA2 → Group3 — L-shaped
d.line([(1630, by+40), (1630, by+190), (100, by+190)], fill=(80,80,80), width=2)
d.polygon([(100, by+190), (100+12, by+190-6), (100+12, by+190+6)], fill=(80,80,80))
# Arrow Group3 → DG-CCA3
harr(480, 530, by+190, (100,100,100))
# Depth injection into DG-CCA3
varr(1300, 285, by+145, (80,140,80), 2)

# ========================
# DECODER + OUTPUT
# ========================
# Decoder (right side)
rbox(1300, by+150, 1600, by+230, (240,248,255), (70,130,180))
txt(1450, by+190, "Decoder", fh, (30,80,140))

# Arrow DG-CCA3 → Decoder
harr(830, 900, by+190, (100,100,100))
# L-shaped from DG-CCA3 to Decoder
d.line([(900, by+190), (900, by+430), (1300, by+430), (1300, by+230)], fill=(80,80,80), width=2)
d.polygon([(1300, by+230), (1300+12, by+230-6), (1300+12, by+230+6)], fill=(80,80,80))

# Output
rbox(1650, by+150, 1900, by+220, (255,235,220), (180,80,40))
txt(1775, by+185, "Restored Î", fh, (140,50,20))
harr(1600, 1650, by+190, (80,80,80))

# Global residual
d.line([(290, by+80), (290, by+500), (1650, by+500), (1650, by+220)], fill=(80,80,80), width=1)
txt(450, by+510, "Global Residual Connection", fs, (80,80,80))

# ========================
# BOTTOM: DG-CCA Detail + Sky TV
# ========================
bx, by2 = 60, 590
rbox(bx, by2, bx+800, by2+420, (253,253,250), (120,120,120), width=1)
txt(bx+400, by2+15, "DG-CCA Detail (per module)", fh, (80,40,80))

# F_rgb → F_d
rbox(bx+50, by2+50, bx+190, by2+85, (220,240,255), (60,100,160))
txt(bx+120, by2+67, "F_rgb", fs, (50,80,140))
rbox(bx+610, by2+50, bx+750, by2+85, (220,255,220), (60,140,60))
txt(bx+680, by2+67, "F_d", fs, (50,120,50))

# GAP
harr(bx+190, bx+240, by2+67)
harr(bx+610, bx+560, by2+67)
varr(bx+240, by2+85, by2+115)
varr(bx+560, by2+85, by2+115)

rbox(bx+210, by2+115, bx+290, by2+148, (230,240,255), (60,100,160))
txt(bx+250, by2+131, "z_rgb", fs, (50,80,140))
rbox(bx+510, by2+115, bx+590, by2+148, (230,255,230), (60,140,60))
txt(bx+550, by2+131, "z_d", fs, (50,120,50))

# Concat
harr(bx+290, bx+330, by2+131)
harr(bx+510, bx+470, by2+131)
varr(bx+400, by2+148, by2+178)
rbox(bx+320, by2+178, bx+480, by2+210, (255,235,245), (160,40,90))
txt(bx+400, by2+194, "Concat [z_rgb; z_d]", fs, (150,30,80))

# MLP
varr(bx+400, by2+210, by2+243)
rbox(bx+310, by2+243, bx+490, by2+275, (255,225,235), (160,30,80))
txt(bx+400, by2+259, "FC→ReLU→FC→σ → A", fs, (150,20,70))

# Multiply out
harr(bx+490, bx+550, by2+259, (160,30,80))
txt(bx+620, by2+259, "F_out = A ⊙ F_rgb", fn, (140,20,70), "lm")
txt(bx+400, by2+320, "C=64, ~8.5K params/module, 3 modules = 25.5K total", fs, (100,100,100))

# ========================
# SKY TV (Right side)
# ========================
sx, sy2 = 920, 590
rbox(sx, sy2, sx+1020, sy2+420, (255,255,238), (130,130,80), width=1)
txt(sx+510, sy2+18, "Targeted Sky TV Fine-Tuning (Epoch 101–120)", fh, (120,100,30))

rbox(sx+80, sy2+60, sx+260, sy2+95, (255,245,220), (160,100,40))
txt(sx+170, sy2+77, "Depth Mask M (τ=0.1)", fs, (140,80,30))

rbox(sx+80, sy2+115, sx+260, sy2+175, (235,245,255), (80,120,180))
txt(sx+170, sy2+135, "Sky Region", fs, (60,80,140))
txt(sx+170, sy2+155, "(D ≥ τ→mask=1)", fs, (80,100,160))

# TV Loss
rbox(sx+460, sy2+60, sx+700, sy2+95, (255,240,225), (160,80,40))
txt(sx+580, sy2+77, "Spatial TV Loss", fs, (140,60,30))

# Formula
txt(sx+510, sy2+155, "L_TV = Σ (1−M) · ||∇Î||₁", fn, (120,80,30))
txt(sx+510, sy2+200, "L_rec = L1 + 0.05·L_edge", fs, (80,80,80))
txt(sx+510, sy2+235, "L_total = L_rec + λ·L_TV  (λ=0.01)", fh, (140,30,30))

# Depth threshold visualization
d.line([(sx+80, sy2+360), (sx+800, sy2+360)], fill=(80,80,80), width=1)
d.line([(sx+250, sy2+348), (sx+250, sy2+372)], fill=(200,50,50), width=2)
txt(sx+250, sy2+385, "τ=0.1", fs, (200,50,50))
for x in range(sx+250, sx+800, 2):
    d.point((x, sy2+360-4), fill=(100,150,255))

# ===== SAVE =====
fp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper", "CVPR_2026_Submission_Template", "figs", "architecture_C_tshape.png")
os.makedirs(os.path.dirname(fp), exist_ok=True)
img.save(fp)
print(f"Saved: {fp}")