from src.domain.validation import validation_yolo


def test_validation_yolo():
    assert (
        validation_yolo(
            "/home/sumbalkhan12/Test/barcode-detecion/runs/detect/train2/weights/best.pt",
            "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/yolo_test/images/img_90000.png",
        )
        == "Testing Successful"
    )
