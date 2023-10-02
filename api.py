from fastapi import FastAPI, HTTPException, File, UploadFile
from ultralytics import YOLO
import uvicorn
import os

from models import (
    TestRequest,
    TrainRequest
)

from test_model import test_yolo
from train_model import train_yolo

app = FastAPI()

@app.post("/train_model/")
async def train_model(train_data: TrainRequest):
    try:
        train_yolo(
            train_data.pre_trained_model_path,
            train_data.dataset_config_path,
            train_data.epochs,
        )
        return {"message": "Training completed successfully"}
    except Exception as exp:
        raise HTTPException(
            status_code=500, detail=f"Error during training: {str(exp)}"
        ) from exp

@app.post("/test_model/")
async def test_model(
    path_to_best_weights: UploadFile = File(...),
    path_to_test_data: UploadFile = File(...),
):
    try:
        best_weights_path:str = os.path.join("/tmp", path_to_best_weights.filename)
        test_data_path:str = os.path.join("/tmp", path_to_test_data.filename)

        with open(best_weights_path, "wb") as best_weights_file:
            best_weights_file.write(path_to_best_weights.file.read())
        with open(test_data_path, "wb") as test_data_file:
            test_data_file.write(path_to_test_data.file.read())

        model:YOLO = YOLO(best_weights_path)
        results:str = model(test_data_path)

        bounding_boxes:int = []
        for result in results:
            bounding_boxes.extend(result.boxes.xyxy.tolist())

        return {"bounding_boxes": bounding_boxes, "message": "Testing done successfully"}
    except Exception as exp:
        raise HTTPException(
            status_code=500, detail=f"Error during testing: {str(exp)}"
        ) from exp

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
