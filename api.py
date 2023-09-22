from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from test_model import test_yolo
from train_model import train_yolo


app = FastAPI()


class TrainRequest(BaseModel):
    pre_trained_model_path: str
    dataset_config_path: str
    epochs: int


class TestRequest(BaseModel):
    path_to_best_weights: str
    path_to_test_data: str


@app.post("/train/")
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


@app.post("/test/")
def test_model(test_data: TestRequest):
    try:
        test_yolo(test_data.path_to_best_weights, test_data.path_to_test_data)
        return {"message": "Testing completed successfully"}
    except Exception as exp:
        raise HTTPException(
            status_code=500, detail=f"Error during testing: {str(exp)}"
        ) from exp


if __name__ == "__main__":


    uvicorn.run(app, host="0.0.0.0", port=9000)
