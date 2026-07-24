"""FastAPI service for AquaMind plankton classification."""

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from ml_model.inference import PlanktonClassifier

app = FastAPI(title="AquaMind API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

try:
    classifier = PlanktonClassifier()
    model_error = None
except (FileNotFoundError, ValueError, RuntimeError) as error:
    classifier = None
    model_error = str(error)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ready" if classifier else "unavailable", "detail": model_error}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if classifier is None:
        raise HTTPException(status_code=503, detail=model_error or "Model is unavailable.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    try:
        content = await file.read()
        with Image.open(io.BytesIO(content)) as image:
            return classifier.predict(image)
    except (OSError, UnidentifiedImageError) as error:
        raise HTTPException(status_code=400, detail="The upload is not a valid image.") from error
    finally:
        await file.close()
