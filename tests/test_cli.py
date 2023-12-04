from typer.testing import CliRunner
from src.cli import app

run = CliRunner()


def test_train_cli():
    results = run.invoke(
        app,
        [
            "train",
            "--pre-trained-model-path",
            "/home/sumbalkhan12/Test/barcode-detecion/yolov8n.pt",
            "--dataset-config-path",
            "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/dataset_config.yaml",
            "--epochs",
            str(1),
        ],
    )
    print("Exit Code:", results.exit_code)
    print("Output:", results.output)
    assert results.exit_code == 0
    assert "Successfully trained YOLO model" in results.output


def test_validate_cli():
    results2 = run.invoke(
        app,
        [
            "validate",
            "--path-to-best-weights",
            "/home/sumbalkhan12/Test/barcode-detecion/runs/detect/train2/weights/best.pt",
            "--path-to-test-data",
            "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/yolo_test/images/img_90000.png",
        ],
    )
    print("Exit Code:", results2.exit_code)
    print("Output:", results2.output)
    assert results2.exit_code == 0
    assert "Testing done successfully" in results2.output
