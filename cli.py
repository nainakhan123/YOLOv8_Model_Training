from test import test_yolo
import typer
from train import train_yolo


app = typer.Typer()


@app.command()
def train(
    pre_trained_model_path: str = typer.Argument(),
    data_to_be_trained_path: str = typer.Argument(),
    epochs: int = typer.Option(),
):
    train_yolo(pre_trained_model_path, data_to_be_trained_path, epochs)


@app.command()
def test(
    path_to_best_weights: str = typer.Argument(),
    path_to_test_file: str = typer.Argument(),
):
    test_yolo(path_to_best_weights, path_to_test_file)


if __name__ == "__main__":
    app()
