import io
import json
from pathlib import Path

from PIL import Image

from . import mosaic_core as mc


# ------------------------------------------------------------
# 1. Пути к пресету и тайлам
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PRESET_PATH = BASE_DIR / "preset.json"   # сюда кладёшь свой lego_mosaic_best_preset.json
TILES_DIR = BASE_DIR / "tiles"           # здесь лежат 5 PNG тайлов RGB (...).png


# ------------------------------------------------------------
# 2. Грузим пресет один раз при старте сервера
# ------------------------------------------------------------

if not PRESET_PATH.exists():
    raise FileNotFoundError(f"Preset file not found: {PRESET_PATH}")

with open(PRESET_PATH, "r", encoding="utf-8") as f:
    PRESET = json.load(f)


# ------------------------------------------------------------
# 3. Основная функция для FastAPI
# ------------------------------------------------------------

def generate_3d_mosaic(image_bytes: bytes, size: int) -> Image.Image:
    """
    Главный вход для backend.main:

    - image_bytes: байты загруженного файла (из Form / UploadFile)
    - size: 64 или 96 (S / L)

    Возвращает PIL.Image с готовой 3D-мозаикой.
    """

    # 1) читаем изображение из байтов
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")

    # 2) строим 3D мозаику через mosaic_core.make_3d_mosaic
    mosaic = mc.make_3d_mosaic(
        image=img,
        preset=PRESET,
        size=size,
        tiles_folder=str(TILES_DIR),
    )

    return mosaic


def generate_3d_mosaic_png_bytes(image_bytes: bytes, size: int) -> bytes:
    """
    Удобный хелпер: сразу возвращает PNG-байты для ответа FastAPI.
    """

    img = generate_3d_mosaic(image_bytes, size)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
