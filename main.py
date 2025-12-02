import io

from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
from typing import List, Optional


app = FastAPI(title="YOLO Object Detection API")

model = YOLO("yolov8n.pt")


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

@app.post("/predict/class")
async def predict_class(
    image: UploadFile = File(...),
    target_class: Optional[str] = None
):

    try:
        image_bytes = await image.read()
        image_read = Image.open(io.BytesIO(image_bytes))
        image_read = image_read.convert('RGB')
        
        if target_class:
            class_names = model.names
            class_id = None
            
            for id, name in class_names.items():
                if name.lower() == target_class.lower():
                    class_id = id
                    break
            
            if class_id is None:
                available_classes = list(class_names.values())
                return {
                    "success": False,
                    "error": f"Класс '{target_class}' не найден. Доступные классы: {available_classes}",
                    "available_classes": available_classes
                }
            
            results = model(image_read, classes=[class_id])
        else:
            results = model(image_read)
        
        objects = []
        class_count = 0
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    class_name = r.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    
                    objects.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": bbox
                    })
                    
                    if target_class and class_name.lower() == target_class.lower():
                        class_count += 1
        
        return {
            "success": True,
            "filename": image.filename,
            "target_class": target_class,
            "detections": objects,
            "total_count": len(objects),
            "target_class_count": class_count if target_class else len(objects)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/classes")
async def get_classes():
    try:
        class_names = model.names
        return {
            "success": True,
            "classes": class_names,
            "class_list": list(class_names.values()),
            "total_classes": len(class_names)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }