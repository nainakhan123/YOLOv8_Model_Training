from fastapi.testclient import TestClient
from fastapi import Request
from fastapi.templating import Jinja2Templates
from src.api import app
client = TestClient(app)

templates = Jinja2Templates(directory="src/templates")

def test_train_model(request:Request):
    pretrained_model_path = "/home/sumbalkhan12/Test/barcode-detecion/yolov8n.pt"
    dataset_config_path = "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/dataset_config.yaml"
    epochs = 1

    response = client.post(
        "/train_model/",
        data={"epochs": epochs},
        files={
            "pretrained_model": ("yolov8n.pt", open(pretrained_model_path, "rb")),
            "dataset_config": ("dataset_config.yaml", open(dataset_config_path, "rb")),
        },
    )

    # assert "training_completed.html" in response.content.decode()


def test_validate_model(request:Request):
    test_image_path = "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/yolo_test/images/img_90000.png" 
    with open(test_image_path, "rb") as test_image_file:
        response = client.post(
            "/validation/",
            files={"path_to_test_image": ("img_90000.png", test_image_file)},
        )

    # assert "results.html" in response.content.decode()
    print(response.content.decode())

