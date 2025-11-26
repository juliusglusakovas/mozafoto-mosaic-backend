import json
import io
from PIL import Image
from . import mosaic_core as mc


# -----------------------------
# Load preset (exactly like Gradio)
# -----------------------------
with open("backend/preset.json", "r") as f:
    PRESET_JSON = json.load(f)

PRESET = mc.Preset(
    blur=PRESET_JSON["blur"],
    unsharp_radius=PRESET_JSON["unsharp_radius"],
    unsharp_percent=PRESET_JSON["unsharp_percent"],
    contrast=PRESET_JSON["contrast"],
    brightness=PRESET_JSON["brightness"],
    gamma=PRESET_JSON["gamma"],
)


# -----------------------------
# Load tiles (your 3D blocks)
# -----------------------------
TILES = mc.load_tile_images("backend/tiles")


# -----------------------------
# API wrapper doing EXACT SAME STEPS as GRADIO
# -----------------------------
def generate_3d_mosaic(img_bytes: bytes, size: str):
    # decode uploaded image
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # convert S / L to pixel size
    if size.upper() == "S":
        target_size = 64
    else:
        target_size = 96

    # ---- MAIN PART ----
    # Gradio calls this EXACT function inside mosaic_core.py
    mosaic_pixels = mc.apply_preset_once(
        img,
        PRESET,
        output_size=target_size,
        max_per_color=1160,       # your real inventory constraint
    )

    # convert pixel matrix to 3D mosaic
    mosaic_3d = mc.render_3d_mosaic(mosaic_pixels, TILES)

    # encode PNG to bytes
    buf = io.BytesIO()
    mosaic_3d.save(buf, format="PNG")
    return buf.getvalue()
