import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from scipy.interpolate import splprep, splev

# Output paths
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
BASE_PNG = os.path.join(STATIC_DIR, "heart_stylish_base.png")
OUTPUT_GIF = os.path.join(STATIC_DIR, "heart_stylish_pump.gif")

# Colors
BG_COLOR = "#9fe6d9"
COLOR_VENTRICLE = "#ff4f6d"
COLOR_VENA = "#0077b6"
COLOR_AORTA = "#d93025"
COLOR_OUTLINE = "#1f1f1f"

def smooth_closed_path(pts, scale=1.0, samples=240, s=0.4):
    pts = np.array(pts, dtype=float) * scale
    tck, _ = splprep([pts[:,0], pts[:,1]], s=s, per=False)
    unew = np.linspace(0, 1, samples)
    x, y = splev(unew, tck)
    return x, y

def draw_base_png():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis('off')
    fig.patch.set_alpha(0.0)
    ax.set_facecolor(BG_COLOR)

    def add_path(anchors, face, z=10, scale=1.0, lw=2):
        x, y = smooth_closed_path(anchors, scale)
        verts = list(zip(x, y))
        codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-1) + [Path.CLOSEPOLY]
        verts.append(verts[0])
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=face, edgecolor=COLOR_OUTLINE, lw=lw, zorder=z)
        ax.add_patch(patch)

    # Heart body
    body = [
        (0.0, -2.6), (1.7, -1.4), (2.2, -0.2), (2.3, 0.8), (1.2, 1.1),
        (0.0, 0.6), (-1.2, 1.1), (-2.3, 0.8), (-2.2, -0.2), (-1.7, -1.4)
    ]
    add_path(body, COLOR_VENTRICLE, z=12, scale=1.0, lw=3)

    # Vessels
    aorta = [(0.2, 0.6), (0.8, 1.4), (1.0, 2.2), (0.2, 2.7), (-1.0, 2.6),
             (-1.4, 2.0), (-1.2, 0.9), (-0.6, 0.5), (0.2, 0.6)]
    vena = [(-1.3, 0.6), (-1.6, 1.4), (-1.8, 2.2), (-2.3, 2.0), (-2.1, 1.1),
            (-1.8, 0.4), (-1.3, 0.6)]
    pulmonary = [(-0.5, 0.6), (-0.9, 1.4), (-0.3, 1.6), (0.9, 1.3), (0.7, 0.5),
                 (-0.1, 0.3), (-0.5, 0.6)]

    add_path(aorta, COLOR_AORTA, z=11, lw=3)
    add_path(vena, COLOR_VENA, z=13, lw=3)
    add_path(pulmonary, COLOR_VENA, z=14, lw=3)

    # Save base PNG (transparent edges, mint background)
    os.makedirs(STATIC_DIR, exist_ok=True)
    fig.savefig(BASE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

def make_pump_gif():
    base = Image.open(BASE_PNG).convert("RGBA")
    W, H = 512, 512
    cx, cy = W//2, H//2 - 20

    frames = []
    total = 50
    for f in range(total):
        beat = 0.0
        if 20 <= f <= 30:
            beat = np.sin((f-20)/10.0 * np.pi)
        scale = 1.0 - 0.08 * beat
        bob = int(6 * np.sin(2*np.pi * f/total))

        canvas = Image.new("RGBA", (W, H), BG_COLOR)
        # Shadow
        shadow = Image.new("RGBA", (W, H), (0,0,0,0))
        draw = ImageDraw.Draw(shadow)
        shadow_color = (31, 61, 58, 60)
        draw.ellipse([cx-120, cy+140, cx+120, cy+170], fill=shadow_color)
        canvas.alpha_composite(shadow)

        # Scale heart
        new_w = int(base.width * scale)
        new_h = int(base.height * scale)
        heart_scaled = base.resize((new_w, new_h), Image.LANCZOS)
        x = cx - new_w//2
        y = cy - new_h//2 + bob
        canvas.alpha_composite(heart_scaled, (x, y))

        frames.append(canvas.convert("P", palette=Image.ADAPTIVE))

    frames[0].save(OUTPUT_GIF, save_all=True, append_images=frames[1:], loop=0, duration=40, optimize=True)

if __name__ == "__main__":
    draw_base_png()
    make_pump_gif()
    print("Saved:", OUTPUT_GIF)
