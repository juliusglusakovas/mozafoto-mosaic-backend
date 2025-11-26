from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .api_core import generate_3d_mosaic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def health():
    return {"ok": True}


@app.post("/mosaic")
async def mosaic(file: UploadFile = File(...), size: str = Form("S")):
    try:
        image_bytes = await file.read()
        png = generate_3d_mosaic(image_bytes, size)
        return Response(png, media_type="image/png")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
