"""方案 B：高集成度折叠式 (The Modular Compact)"""
from PIL import Image, ImageDraw, ImageFont
import math, os

W, H = 1800, 1300
img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)

try:
    ft = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)
    fh = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 17)
    fn = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    fs = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
except:
    ft=fh=fn=fs=ImageFont.load_default()

def rbox(x1,y1,x2,y2,fill,outline=(0,0,0),width=2,radius=8):
    d.rounded_rectangle([x1,y1,x2,y2], radius=radius, fill=fill, outline=outline, width=width)

def txt(x,y,s,font=fn,fill=(0,0,0),anchor="mm"):
    b=d.textbbox((0,0),s,font=font); tw,th=b[2]-b[0],b[3]-b[1]
    if anchor=="mm": d.text((x-tw//2,y-th//2),s,fill=fill,font=font)
    elif anchor=="lm": d.text((x,y-th//2),s,fill=fill,font=font)

def harr(x1,x2,y,c=(80,80,80),w=2):
    d.line([(x1,y),(x2,y)],fill=c,width=w); L=12
    d.polygon([(x2,y),(x2-L,y-6),(x2-L,y+6)],fill=c)

def varr(x,y1,y2,c=(80,80,80),w=2):
    d.line([(x,y1),(x,y2)],fill=c,width=w); L=12
    d.polygon([(x,y2),(x-6,y2-L),(x+6,y2-L)],fill=c)

# ===== TITLE =====
txt(W//2, 30, "Architecture Overview — Modular Compact Design", ft, (0,0,0))

# ========================
# TOP: Frozen DAV2
# ========================
rbox(700, 60, 1100, 110, (220,255,220), (60,130,60))
txt(900, 85, "Frozen DAV2 (24.8M)", fh, (20,100,20))

# Hazy input on top left
rbox(60, 60, 240, 110, (220,235,255), (60,100,180))
txt(150, 85, "Hazy I", fh, (20,60,140))

# Arrow from Hazy to DAV2
harr(240, 380, 85, (80,80,80))
# But we need to turn down — use L-shaped
d.line([(380,85),(380,120),(700,120),(700,85)], fill=(80,140,80), width=2)
d.polygon([(700,85),(700-6,85-12),(700+6,85-12)], fill=(80,140,80))

# ========================
# CENTER: Large dashed box "Depth-Guided Restoration Block ×3"
# ========================
cx, cy = 180, 260
cw, ch = 1440, 520
d.rounded_rectangle([cx, cy, cx+cw, cy+ch], radius=15, fill=(248,250,252), outline=(100,100,140), width=2)
d.rectangle([cx+5, cy-22, cx+350, cy+2], fill=(248,250,252), outline=(100,100,140), width=1)
txt(cx+175, cy-12, "Depth-Guided Restoration Block (×3)", fs, (80,80,120))

# Inside the box: Group + DG-CCA repeated 3 times vertically
for i in range(3):
    iy = cy + 30 + i * 150
    # Group process
    rbox(cx+80, iy, cx+380, iy+70, (240,248,255), (70,130,180))
    txt(cx+230, iy+35, f"Group Process {i+1}", fh, (30,80,140))
    
    # Arrow → DG-CCA
    harr(cx+380, cx+460, iy+35, (100,100,100))
    
    # DG-CCA
    rbox(cx+460, iy-10, cx+680, iy+80, (255,225,235), (200,50,100), width=3)
    txt(cx+570, iy+20, "DG-CCA", fh, (180,20,80))
    txt(cx+570, iy+50, "F_rgb, F_d→GAP→MLP→σ→⊙", fs, (140,30,70))
    
    # Arrow right →
    if i < 2:
        harr(cx+680, cx+720, iy+35)
        # vertical down arrow
        varr(cx+700, iy+70, iy+120, (80,80,80), 1)
    else:
        # Last one: arrow right out
        harr(cx+680, cx+740, iy+35, (80,80,80))

# Vertical depth injection from top down to middle of each DG-CCA
draw_line = d.line
for i in range(3):
    iy = cy + 30 + i * 150
    ix = cx + 570
    # Arrow from top of big box down
    draw_line([(900, 110), (900, 150), (ix, iy-10)], fill=(80,140,80), width=2)
    # Small dot
    d.polygon([(ix, iy-10), (ix-6, iy-20), (ix+6, iy-20)], fill=(80,140,80))

# ========================
# RIGHT: Decoder + Output
# ========================
# Decoder
rbox(cx+cw+30, cy+80, cx+cw+170, cy+150, (240,248,255), (70,130,180))
txt(cx+cw+100, cy+115, "Decoder", fh, (30,80,140))

harr(cx+680, cx+cw+30, cy+115, (80,80,80))

# Output
rbox(cx+cw+30, cy+210, cx+cw+230, cy+270, (255,235,220), (180,80,40))
txt(cx+cw+130, cy+240, "Restored Î", fh, (140,50,20))
varr(cx+cw+100, cy+150, cy+210, (80,80,80))

# ========================
# BOTTOM: DG-CCA Detail + Sky TV Loss
# ========================
bx, by = 200, 830
rbox(bx, by, bx+640, by+430, (253,253,250), (120,120,120), width=1)
txt(bx+320, by+12, "DG-CCA Detail (inside each block)", fh, (80,40,80))

# F_rgb
rbox(bx+50, by+45, bx+180, by+78, (220,240,255), (60,100,160))
txt(bx+115, by+61, "F_rgb", fs, (50,80,140))

# F_d
rbox(bx+460, by+45, bx+590, by+78, (220,255,220), (60,140,60))
txt(bx+525, by+61, "F_d", fs, (50,120,50))

# GAP arrows + boxes
harr(bx+180, bx+230, by+61, (80,80,80))
harr(bx+460, bx+400, by+61, (80,80,80))
varr(bx+230, by+78, by+110, (80,80,80))
varr(bx+400, by+78, by+110, (80,80,80))

rbox(bx+200, by+110, bx+270, by+140, (230,240,255), (60,100,160))
txt(bx+235, by+125, "z_rgb", fs, (50,80,140))
rbox(bx+370, by+110, bx+440, by+140, (230,255,230), (60,140,60))
txt(bx+405, by+125, "z_d", fs, (50,120,50))

# Concat
harr(bx+270, bx+310, by+125, (80,80,80))
harr(bx+370, bx+330, by+125, (80,80,80))
varr(bx+320, by+140, by+170, (80,80,80))
rbox(bx+260, by+170, bx+380, by+200, (255,235,245), (160,40,90))
txt(bx+320, by+185, "Concat", fs, (150,30,80))

# MLP
varr(bx+320, by+200, by+230)
rbox(bx+250, by+230, bx+390, by+260, (255,225,235), (160,30,80))
txt(bx+320, by+245, "FC→ReLU→FC→σ", fs, (150,20,70))

# Multiply
harr(bx+390, bx+430, by+245, (160,30,80))
txt(bx+480, by+245, "F_out = A ⊙ F_rgb", fn, (140,20,70), "lm")

# ========================
# SKY TV LOSS (Right side)
# ========================
sx, sy = 900, 830
rbox(sx, sy, sx+780, sy+430, (255,255,238), (130,130,80), width=1)
txt(sx+390, sy+15, "Targeted Sky TV Fine-Tuning (Epoch 101–120)", fh, (120,100,30))

rbox(sx+60, sy+60, sx+240, sy+95, (255,245,220), (160,100,40))
txt(sx+150, sy+77, "Depth Mask (τ=0.1)", fs, (140,80,30))

# Sky region
rbox(sx+300, sy+60, sx+460, sy+95, (235,245,255), (80,120,180))
txt(sx+380, sy+77, "Sky Region D≥τ", fs, (60,80,140))

# TV loss
rbox(sx+60, sy+120, sx+300, sy+155, (255,240,225), (160,80,40))
txt(sx+180, sy+137, "Spatial TV Loss", fs, (140,60,30))

# Formula
txt(sx+390, sy+185, "L_TV = Σ(1−M)·||∇Î||₁", fn, (120,80,30))
txt(sx+390, sy+225, "L_rec = L1 + 0.05·L_edge", fs, (80,80,80))
txt(sx+390, sy+260, "L_total = L_rec + λ·L_TV (λ=0.01)", fh, (140,30,30))

# Small sky example visualization
# Simple diagram: depth threshold
# Horizontal depth axis
d.line([(sx+60, sy+350), (sx+550, sy+350)], fill=(80,80,80), width=1)
# Threshold line
d.line([(sx+150, sy+340), (sx+150, sy+360)], fill=(200,50,50), width=2)
txt(sx+150, sy+370, "τ=0.1", fs, (200,50,50))
# Shaded sky region
for x in range(sx+150, sx+550, 2):
    d.point((x, sy+350-5), fill=(100,150,255))

# ===== SAVE =====
fp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper", "CVPR_2026_Submission_Template", "figs", "architecture_B_compact.png")
os.makedirs(os.path.dirname(fp), exist_ok=True)
img.save(fp)
print(f"Saved: {fp}")