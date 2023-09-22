from fastapi import FastAPI, HTTPException
from ultralytics import YOLO
import uvicorn

from models import TrainRequest
from models import TestRequest
from test_model import test_yolo
from train_model import train_yolo


app = FastAPI()


@app.post("/train_model/")
def train_model(train_data: TrainRequest):
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
def test_model(test_data: TestRequest):
    try:
        model = YOLO(test_data.path_to_best_weights)
        results = model(test_data.path_to_test_data)

        bounding_boxes = []
        for result in results:
            bounding_boxes.extend(result.boxes.xyxy.tolist())

        return {"bounding_boxes": bounding_boxes,"message": "Testing done successfully" }
    except Exception as exp:
        raise HTTPException(
            status_code=500, detail=f"Error during testing: {str(exp)}"
        ) from exp


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)