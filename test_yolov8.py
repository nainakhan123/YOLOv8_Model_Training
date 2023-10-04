# import pytest
from train_model import train_yolo
from validation import validation_yolo


def test_train_yolo():
    assert (
        train_yolo(
            "/home/sumbalkhan12/Test/barcode-detecion/yolov8n.pt",
            "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/dataset_config.yaml",
            1,
        )
        == "Successfully trained YOLO model"
    )


# @pytest.fixture
# def path_to_best_weights():
# return "/home/sumbalkhan12/Test/barcode-detecion/runs/detect/train2/weights/best.pt"
# return "/home/sumbalkhan12/YOLOv8_Model_Training/runs/detect/train2/weights/best.pt"

# @pytest.fixture
# def path_to_test_data():
#     return "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/yolo_test/images/img_90000.png"


def test_validation_yolo():
    assert (
        validation_yolo(
            "/home/sumbalkhan12/Test/barcode-detecion/runs/detect/train2/weights/best.pt",
            "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/yolo_test/images/img_90000.png",
        )
        == "Testing Successful"
    )
