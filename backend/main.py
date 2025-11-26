from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import Response
from .api_core import generate_3d_mosaic

app = FastAPI()

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/mosaic")
async def mosaic_endpoint(
    file: UploadFile,
    size: str = Form("S")
):
    """
    file: image uploaded by user
    size: "S" (64) or "L" (96)
    """

    img_bytes = await file.read()

    # run your FULL pipeline from mosaic_core.py
    final_img = generate_3d_mosaic(
        img_bytes=img_bytes,
        size=size
    )

    return Response(
        content=final_img,
        media_type="image/png"
    )
