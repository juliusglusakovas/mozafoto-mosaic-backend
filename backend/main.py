# backend/main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .mosaic_logic import generate_3d_mosaic

app = FastAPI(title="Mozafoto Mosaic Backend")


# CORS, чтобы спокойно дергать из app.mozafoto.lt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можешь позже сузить до своего домена
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post(
    "/mosaic",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def mosaic_endpoint(
    file: UploadFile = File(...),
    size: str = Form("S"),  # "S" или "L"
):
    """
    Принимает фото + размер ("S" или "L"),
    возвращает PNG с 3D LEGO мозайкой.
    """
    image_bytes = await file.read()
    size = (size or "S").upper()
    if size not in ("S", "L"):
        size = "S"

    mosaic_png = generate_3d_mosaic(image_bytes, size=size)
    return Response(content=mosaic_png, media_type="image/png")
