from ultralytics import YOLO


def train_yolo(
    pre_trained_model_config_path: str,
    dataset_to_be_trained_path: str,
    epochs: int,
):
    print("start")
    model: YOLO = YOLO(pre_trained_model_config_path)
    print("Successfully loaded YOLO model")
    model.train(data=dataset_to_be_trained_path, epochs=epochs)
    print("Successfully trained YOLO model")
    model.val()
