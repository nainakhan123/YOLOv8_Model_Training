import typer
from ultralytics import YOLO


app = typer.Typer()


@app.command()
def train_yolo(
    pre_trained_model_config_path: str = typer.Argument(),
    dataset_to_be_trained_path: str = typer.Argument(),
    epochs: int = typer.Option(),
):
    print("start")
    model: YOLO = YOLO(pre_trained_model_config_path)
    print("Successfully loaded YOLO model")
    model.train(data=dataset_to_be_trained_path, epochs=epochs)
    print("Successfully trained YOLO model")
    model.val()


if __name__ == "__main__":
    app()
