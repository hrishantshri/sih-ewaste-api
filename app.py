from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
from pathlib import Path


# --------------------------------
# FastAPI App
# --------------------------------

app = FastAPI(
    title="E-Waste Detection API",
    description="YOLO-based e-waste detection API",
    version="1.0.0"
)


# --------------------------------
# CORS
# --------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------
# Model
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "e_waste_final_v3_best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))


# --------------------------------
# Health Check
# --------------------------------

@app.get("/")
def root():
    return {
        "success": True,
        "message": "E-Waste Detection API is running",
        "model": "YOLO26n",
        "classes": model.names
    }


# --------------------------------
# Prediction API
# --------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    try:

        # Read uploaded image
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # YOLO prediction
        results = model.predict(
            source=image,
            conf=0.50,
            imgsz=640,
            verbose=False
        )

        detections = []

        result = results[0]

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "name": model.names[class_id],
                    "confidence": round(confidence, 3),
                    "box": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2)
                    ]
                })

        return {
            "success": True,
            "filename": file.filename,
            "detections": detections,
            "count": len(detections)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
