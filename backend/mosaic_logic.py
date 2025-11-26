# backend/mosaic_logic.py

import io
import os
from functools import lru_cache

import numpy as np
from PIL import Image

# Размер одного LEGO-тайла в итоговой мозаике (в пикселях)
BASE_TILE_SIZE = 32

# Палитра из твоей GradIO апки (от светлого к тёмному)
PALETTE = np.array(
    [
        [220, 224, 225],  # индекс 0
        [150, 160, 171],  # индекс 1
        [88, 99, 110],    # индекс 2
        [35, 46, 59],     # индекс 3
        [20, 25, 30],     # индекс 4
    ],
    dtype=np.float32,
)

# Соответствие индекса палитры файлу PNG
TILE_FILENAMES = [
    "RGB (220, 224, 225).png",
    "RGB (150, 160, 171).png",
    "RGB (88, 99, 110).png",
    "RGB (35, 46, 59).png",
    "RGB (20, 25, 30).png",
]


@lru_cache(maxsize=1)
def load_tiles():
    """
    Загружаем 5 PNG-тайлов и приводим к размеру BASE_TILE_SIZE x BASE_TILE_SIZE.
    Кэшируем, чтобы не читать с диска каждый запрос.
    """
    base_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(base_dir, ".."))

    tiles = []
    for name in TILE_FILENAMES:
        path = os.path.join(repo_root, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tile image not found: {path}")
        img = Image.open(path).convert("RGB")
        if img.size != (BASE_TILE_SIZE, BASE_TILE_SIZE):
            img = img.resize((BASE_TILE_SIZE, BASE_TILE_SIZE), Image.LANCZOS)
        tiles.append(img)

    return tiles


def _center_square_crop(img: Image.Image) -> Image.Image:
    """Обрезка до центрального квадрата (как в твоём UI 1:1)."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _quantize_to_palette(img: Image.Image) -> np.ndarray:
    """
    Принимает маленькое изображение (NxN), возвращает матрицу индексов палитры (N x N),
    где каждое значение 0..4 — индекс ближайшего цвета из PALETTE.
    """
    arr = np.asarray(img).astype(np.float32)  # (H, W, 3)
    h, w, _ = arr.shape

    pixels = arr.reshape(-1, 3)[:, None, :]  # (H*W, 1, 3)
    palette = PALETTE[None, :, :]            # (1, 5, 3)

    # квадраты расстояний до каждого цвета палитры
    d2 = ((pixels - palette) ** 2).sum(axis=2)  # (H*W, 5)

    indices = d2.argmin(axis=1).reshape(h, w).astype(np.int32)
    return indices


def generate_3d_mosaic(image_bytes: bytes, size: str = "S") -> bytes:
    """
    Главная функция:
    - принимает байты исходного изображения
    - size = "S" (64x64) или "L" (96x96)
    - возвращает PNG-байты 3D-моцаики (с LEGO-тайлами)
    """

    # 1. читаем картинку
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2. квадратная обрезка
    img = _center_square_crop(img)

    # 3. выбираем размер мозаики
    size = (size or "S").upper()
    if size == "L":
        mosaic_n = 96
    else:
        mosaic_n = 64

    # 4. уменьшаем до сетки мозаики
    small = img.resize((mosaic_n, mosaic_n), Image.LANCZOS)

    # 5. квантуем в 5 цветов
    indices = _quantize_to_palette(small)  # (N, N), значения 0..4

    # 6. загружаем тайлы
    tiles = load_tiles()
    tile_w, tile_h = tiles[0].size  # (BASE_TILE_SIZE, BASE_TILE_SIZE)

    # 7. собираем финальную 3D-мозаику
    out_w, out_h = mosaic_n * tile_w, mosaic_n * tile_h
    out = Image.new("RGB", (out_w, out_h))

    for y in range(mosaic_n):
        for x in range(mosaic_n):
            idx = int(indices[y, x])
            tile = tiles[idx]
            out.paste(tile, (x * tile_w, y * tile_h))

    # 8. кодируем в PNG
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
