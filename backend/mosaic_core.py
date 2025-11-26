import os
import json
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ------------------------------------------------------------
# 1. LOAD TILES
# ------------------------------------------------------------
def load_tile_images_from_folder(folder):
    """
    Loads all RGB tile PNGs (e.g. 'RGB (20, 25, 30).png') from folder.
    Returns dict: { (r,g,b): PIL.Image }
    """
    tiles = {}
    for filename in os.listdir(folder):
        if not filename.lower().endswith(".png"):
            continue

        # filename format: RGB (r, g, b).png
        name = filename.replace("RGB", "").replace("(", "").replace(")", "")
        name = name.replace(".png", "").strip()
        r, g, b = map(int, name.split(","))

        img = Image.open(os.path.join(folder, filename)).convert("RGBA")
        tiles[(r, g, b)] = img

    return tiles


# ------------------------------------------------------------
# 2. APPLY PRESET (IDENTICAL to GRADIO version)
# ------------------------------------------------------------
def apply_preset_once(image, preset, output_size, max_per_color=9999999):
    """
    This is EXACTLY the same transformation as your Gradio app.
    1) Resize
    2) Blur → Sharpen → Contrast → Brightness → Gamma
    3) Quantize to preset colors
    """

    # --- Resize ---
    image = image.resize((output_size, output_size), Image.Resampling.LANCZOS)

    # --- Apply filters in EXACT order ---
    if preset.get("blur", 0) > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=preset["blur"]))

    if preset.get("sharpen", 0) > 0:
        for _ in range(preset["sharpen"]):
            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))

    if preset.get("contrast", 1) != 1:
        image = ImageEnhance.Contrast(image).enhance(preset["contrast"])

    if preset.get("brightness", 1) != 1:
        image = ImageEnhance.Brightness(image).enhance(preset["brightness"])

    if preset.get("gamma", 1) != 1:
        arr = np.array(image).astype(np.float32) / 255.0
        arr = np.power(arr, preset["gamma"])
        image = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))

    # --- Quantize to 5 preset colors ---
    colors = preset["colors"]  # list of RGB colors
    img_arr = np.array(image).astype(np.float32)

    h, w, _ = img_arr.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            pixel = img_arr[y, x]
            # find nearest preset color
            diffs = np.sum((colors - pixel) ** 2, axis=1)
            idx = np.argmin(diffs)
            out[y, x] = colors[idx]

    return out  # matrix of shape (H,W,3) with EXACT preset colors


# ------------------------------------------------------------
# 3. RENDER 3D MOSAIC (IDENTICAL to GRADIO)
# ------------------------------------------------------------
def render_3d_mosaic(pixel_matrix, tiles):
    """
    PLACE THE 3D TILE PNG on each pixel.
    This is EXACT reproduction of your original function.
    """

    h, w, _ = pixel_matrix.shape

    # Size of a single tile
    example_tile = next(iter(tiles.values()))
    tw, th = example_tile.size

    # Output image
    out = Image.new("RGBA", (w * tw, h * th), (0, 0, 0, 0))

    for y in range(h):
        for x in range(w):
            color = tuple(pixel_matrix[y, x])
            tile = tiles.get(color)

            if tile is None:
                # fallback: choose nearest tile
                best_key = min(
                    tiles.keys(),
                    key=lambda c: (c[0] - color[0]) ** 2 + (c[1] - color[1]) ** 2 + (c[2] - color[2]) ** 2
                )
                tile = tiles[best_key]

            out.paste(tile, (x * tw, y * th), tile)

    return out


# ------------------------------------------------------------
# 4. MAIN FUNCTION FOR API
# ------------------------------------------------------------
def make_3d_mosaic(image: Image.Image, preset: dict, size: int, tiles_folder: str):
    """
    This is the simplified "one-call" function used by your API:
        1. load tiles
        2. apply preset
        3. render 3D mosaic
    """

    # load tile images once
    tiles = load_tile_images_from_folder(tiles_folder)

    # apply preset to get pixel grid
    pixels = apply_preset_once(image, preset, output_size=size)

    # build 3d lego mosaic
    mosaic = render_3d_mosaic(pixels, tiles)

    return mosaic
