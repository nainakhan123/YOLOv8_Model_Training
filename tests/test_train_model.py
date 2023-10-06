# import os
# import pytest
import sys
sys.path.append("/home/sumbalkhan12/YOLOv8_Model_Training/yolov8/")
from train_model import train_yolo
# from .config import pre_trained_model_path, dataset_config_path, epochs


# @pytest.fixture
# def pre_trained_model_path():
#     return "/home/sumbalkhan12/YOLOv8_Model_Training/yolov8/yolov8n.pt"


# @pytest.fixture
# def dataset_config_path():
#     return "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/dataset_config.yaml"


# @pytest.fixture
# def epochs():
#     return 1


def test_train_yolo():
    assert (
        train_yolo("/home/sumbalkhan12/Test/barcode-detecion/yolov8n.pt", "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/dataset_config.yaml", 1)
        == "Successfully trained YOLO model"
    )
