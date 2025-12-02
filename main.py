import io

from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from pydantic import BaseModel
from PIL import Image


app = FastAPI(title="YOLO Object Detection API")

model = YOLO("yolov8n.pt")


class ImageRequest(BaseModel):
    image_base64: str
    confidence_threshold: float = 0.5
    

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    box: list


@app.get("/")
def home():
    return {"message": "YOLO Object Detection API", "status": "работает"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        
        image_read = Image.open(io.BytesIO(image_bytes))
        
        image_read = image_read.convert('RGB')
        
        results = model(image_read)
        
        objects = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    objects.append({
                        "class": r.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist()
                    })
        
        return {
            "success": True,
            "filename": image.filename,
            "detections": objects,
            "count": len(objects)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }