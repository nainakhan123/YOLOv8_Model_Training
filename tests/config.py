import pytest

@pytest.fixture
def pre_trained_model_path():
    return "/home/sumbalkhan12/Test/barcode-detecion/"


@pytest.fixture
def dataset_config_path():
    return "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/dataset_config.yaml"


@pytest.fixture
def epochs():
    return 1


@pytest.fixture
def path_to_best_weights():
    return "/home/sumbalkhan12/Test/barcode-detecion/runs/detect/train2/weights/best.pt"


@pytest.fixture
def path_to_test_data():
    return "/home/sumbalkhan12/Test/barcode-detecion/test_dataset/yolo_test/images/img_90000.png"