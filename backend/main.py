from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import Response
from .api_core import generate_3d_mosaic

app = FastAPI()

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/mosaic")
async def mosaic(
    file: UploadFile,
    size: str = Form("S")
):
    img_bytes = await file.read()

    result = generate_3d_mosaic(
        img_bytes=img_bytes,
        size=size
    )

    return Response(content=result, media_type="image/png")
