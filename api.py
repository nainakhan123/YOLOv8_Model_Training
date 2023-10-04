import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from ultralytics import YOLO
import uvicorn

from models import TrainRequest


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


# best_weights_path = "/home/sumbalkhan12/YOLOv8_Model_Training/runs/detect/train2/weights/best.pt"
best_weights_path = "/home/sumbalkhan12/Test/barcode-detecion/runs/detect/train2/weights/best.pt"


@app.post("/validation/")
async def validation(
    path_to_test_image: UploadFile = File(...),
):
    try:
        test_image_path: str = os.path.join("/tmp", path_to_test_image.filename)

        with open(test_image_path, "wb") as test_image_file:
            test_image_file.write(path_to_test_image.file.read())
            print("picture uploaded")

        model = YOLO(best_weights_path)
        results = model(test_image_path)
        print("Results")
        bounding_boxes = []
        for result in results:
            bounding_boxes.extend(result.boxes.xyxy.tolist())

        return {
            "bounding_boxes": bounding_boxes,
            "message": "Testing done successfully",
        }
    except Exception as exp:
        raise HTTPException(
            status_code=500, detail=f"Error during testing: {str(exp)}"
        ) from exp
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
