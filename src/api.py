import os
import io
import traceback
import base64
import boto3

import uvicorn
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw


from src.domain.train_model import train_yolo
from src.aws_s3_upload import create_upload_file

app=FastAPI()

s3=boto3.client('s3')

bucket_name = 'barcodemlmodels'
best_weights_key = 'model_new/best.pt'
local_best_weights_path = '/tmp/best.pt'

s3.download_file(bucket_name, best_weights_key, local_best_weights_path)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/train_model/", response_class=HTMLResponse)
async def read_train(request: Request): 
    return templates.TemplateResponse("train_page.html", {"request": request})


@app.post("/train_model/")
async def train_model(
    request: Request,
    pretrained_model: UploadFile = File(...),
    dataset_config: UploadFile = File(...),
    epochs: int = Form(...),
):
    try:
        pretrained_model_bytes = await pretrained_model.read()
        dataset_config_bytes = await dataset_config.read()

        pretrained_model_path = "/tmp/pretrained_model.pt"
        dataset_config_path = "/tmp/dataset_config.yaml"

        with open(pretrained_model_path, "wb") as pretrained_model_file:
            pretrained_model_file.write(pretrained_model_bytes)

        with open(dataset_config_path, "wb") as dataset_config_file:
            dataset_config_file.write(dataset_config_bytes)

        train_yolo(
            pretrained_model_path,
            dataset_config_path,
            epochs,
        )

        return templates.TemplateResponse(
            "training_completed.html", {"request": request}
        )
    except Exception as exp:
        raise HTTPException(
            status_code=500, detail=f"Error during training: {str(exp)}"
        ) from exp


@app.get("/validation/", response_class=HTMLResponse)
async def read_validate(request: Request):
    return templates.TemplateResponse("validate_page.html", {"request": request})


@app.post("/validation/")
async def validation(
    request: Request,
    path_to_test_image: UploadFile = File(...),
):
    try:
        test_image_path: str = os.path.join("/tmp", path_to_test_image.filename)
        # best_weights_path ="src/best.pt"
    #  "/home/sumbalkhan12/Test/barcode-detection/runs/detect/train2/weights/best.pt"
        with open(test_image_path, "wb") as test_image_file:
            test_image_file.write(path_to_test_image.file.read())
            print("picture uploaded")

        model = YOLO(local_best_weights_path)
        results = model(test_image_path)
        print("Results")
        bounding_boxes = []

        image = Image.open(test_image_path)
        draw = ImageDraw.Draw(image)

        for result in results:
            boxes = result.boxes.xyxy.tolist()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle([x1, y1, x2, y2], width=4)
                bounding_boxes.append(box)

        if len(bounding_boxes) > 0:
            message = "Barcode Detected"
        else:
            message = "Barcode Not Detected"

        output_image_buffer = io.BytesIO()
        image.save(output_image_buffer, format="PNG")
        output_image_buffer.seek(0)

        output_image = base64.b64encode(output_image_buffer.read()).decode()

        create_upload_file(path_to_test_image)

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "message": message,
                "output_image": output_image,
                "bounding_boxes": boxes,
            },
        )
    except Exception as exp:
        traceback.print_exc()
        print(f"Error during validation: {str(exp)}")
        raise HTTPException(
            status_code=500, detail=f"Error during testing: {str(exp)}"
        ) from exp


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")