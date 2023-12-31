from ultralytics import YOLO


def train_yolo(
    pre_trained_model_path: str,
    dataset_config_path: str,
    epochs: int,
):
    # print("start")
    model: YOLO = YOLO(pre_trained_model_path)
    print("Successfully loaded YOLO model")
    model.train(data=dataset_config_path, epochs=epochs)
    print("Successfully trained YOLO model")
    model.val()
    return "Successfully trained YOLO model"
