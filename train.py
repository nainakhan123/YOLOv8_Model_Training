import typer
from ultralytics import YOLO


app = typer.Typer()


@app.command()
def train_yolo(
    pre_trained_model_path: str = typer.Argument(),
    data_to_be_trained_path: str = typer.Argument(),
    epochs: int = typer.Option(),
):
    print("start")
    model: YOLO = YOLO(pre_trained_model_path)
    print("Successfully loaded YOLO model")
    model.train(data=data_to_be_trained_path, epochs=epochs)
    print("Successfully trained YOLO model")
    model.val()


if __name__ == "__main__":
    app()
