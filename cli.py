import typer

from test import test_yolo
from train import train_yolo


app = typer.Typer()


@app.command()
def train(
    pre_trained_model_path: str = typer.Option(
        "--path_to_pretrained_model", help="Path to pre-trained model"
    ),
    data_to_be_trained_config_path: str = typer.Option(
        "--data_to_be_trained_config_path", help="Path to dataset to be train"
    ),
    epochs: int = typer.Option("--epochs", help="Number of epochs"),
):
    train_yolo(pre_trained_model_path, data_to_be_trained_config_path, epochs)


@app.command()
def test(
    path_to_best_weights: str = typer.Option(
        "--best_weight_path", help="Path to weights generated after training"
    ),
    path_to_test_data: str = typer.Option(
        "--test_data_path", help="Path to test dataset it could be a directory or a single picture"
    ),
):
    test_yolo(path_to_best_weights, path_to_test_data)


if __name__ == "__main__":
    app()
