import json
import io
from PIL import Image
from . import mosaic_core as mc


# -----------------------------
# Load preset (same as Gradio)
# -----------------------------
with open("preset.json", "r") as f:
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
# Load 3D LEGO tiles
# Your mosaic_core.py loads them like this:
# -----------------------------
TILES = mc.load_tile_images_from_folder("tiles")


# -----------------------------
# Full pipeline identical to Gradio
# -----------------------------
def generate_3d_mosaic(img_bytes: bytes, size: str):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # map S/L size to pixels
    target_size = 64 if size.upper() == "S" else 96

    # this is EXACTLY what your Gradio uses:
    pixels = mc.apply_preset_once(
        img,
        PRESET,
        output_size=target_size,
        max_per_color=1160
    )

    # 3D rendering
    mosaic = mc.render_3d_mosaic(pixels, TILES)

    buf = io.BytesIO()
    mosaic.save(buf, format="PNG")
    return buf.getvalue()
