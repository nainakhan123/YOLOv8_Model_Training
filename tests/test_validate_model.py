import sys
sys.path.append("/home/sumbalkhan12/YOLOv8_Model_Training/yolov8/")
from validation import validation_yolo
# from .config import (
#     path_to_best_weights,
#     path_to_test_data
# )

def test_validation_yolo():
    assert (
    validation_yolo(
            "/home/sumbalkhan12/Test/barcode-detecion/runs/detect/train2/weights/best.pt",
            "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/yolo_test/images/img_90000.png",
        )
        == "Testing Successful"
    )
