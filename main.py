from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import shutil
import uuid
import os
import cv2
from typing import List

app = FastAPI()
model = YOLO("best.pt")

# Folder to save annotated images temporarily
ANNOTATED_FOLDER = "annotated_images"
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file
    temp_filename = f"{uuid.uuid4()}.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    results = model(temp_filename)
    result = results[0]
    
    # Draw results
    annotated_image = result.plot()

    # Save annotated image
    annotated_filename = f"{uuid.uuid4()}.jpg"
    annotated_path = os.path.join(ANNOTATED_FOLDER, annotated_filename)
    cv2.imwrite(annotated_path, annotated_image)

    # Extract prediction data
    predictions: List[dict] = []
    for box in result.boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = box
        predictions.append({
            "class_id": int(class_id),
            "class_name": model.names[int(class_id)],
            "confidence": float(conf),
            "bbox": [x1, y1, x2, y2]
        })

    # Clean up uploaded temp file
    os.remove(temp_filename)

    # Return JSON with image path
    return JSONResponse({
        "predictions": predictions,
        "image_url": f"/image/{annotated_filename}"
    })


@app.get("/image/{filename}")
def get_annotated_image(filename: str):
    image_path = os.path.join(ANNOTATED_FOLDER, filename)
    if not os.path.exists(image_path):
        return JSONResponse({"error": "Image not found"}, status_code=404)

    return StreamingResponse(open(image_path, "rb"), media_type="image/jpeg")
