from io import BytesIO
from PIL import Image
import mosaic_core  # tavo tikras kodas bus čia


def generate_3d_mosaic(image_bytes: bytes, size: str):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # ČIA TIK PAKEISK i tavo tikrą funkciją
    if hasattr(mosaic_core, "make_mosaic_3d"):
        result = mosaic_core.make_mosaic_3d(img, size=size)
    else:
        result = img

    out = BytesIO()
    result.save(out, format="PNG")
    return out.getvalue()
