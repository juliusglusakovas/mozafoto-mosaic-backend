import io
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import Response
from PIL import Image
from .api_core import generate_3d_mosaic_png_bytes  # ← импорт хелпера, который возвращает БАЙТЫ

app = FastAPI()

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/mosaic")
async def mosaic(
    file: UploadFile = File(...),
    size: str = Form("S")  # "S" или "L"
):
    # читаем оригинальные байты
    img_bytes = await file.read()

    # выбираем размер в PIXELS
    if size.upper() == "S":
        px = 64
    elif size.upper() == "L":
        px = 96
    else:
        px = 64  # fallback, если пришло что-то странное

    # получаем PNG-байты уже ГОТОВОЙ 3D мозаики
    result_bytes = generate_3d_mosaic_png_bytes(img_bytes, px)

    return Response(content=result_bytes, media_type="image/png")
