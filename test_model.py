from ultralytics import YOLO


def test_yolo(path_to_best_weights: str, path_to_test_data: str):
    print("Testing started")
    model: YOLO = YOLO(path_to_best_weights)
    model.predict(path_to_test_data, save=True)
    print("Testing done successfully")
