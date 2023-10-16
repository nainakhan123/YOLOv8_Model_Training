from src.domain.train_model import train_yolo


def test_train_yolo():
    assert (
        train_yolo(
            "/home/sumbalkhan12/Test/barcode-detecion/yolov8n.pt",
            "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/dataset_config.yaml",
            1,
        )
        == "Successfully trained YOLO model"
    )
