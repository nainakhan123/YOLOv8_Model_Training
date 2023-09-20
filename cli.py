from test import test_yolo
import typer
from train import train_yolo


app = typer.Typer()


@app.command()
def train(
    pre_trained_model_path: str = typer.Option(
        "--path_to_pretrained_model", help="Path to pre-trained model"
    ),
    data_to_be_trained_path: str = typer.Option(
        "--path_to_dataset_to_be_trained", help="Path to dataset to be train"
    ),
    epochs: int = typer.Option("--epochs", help="Number of epochs"),
):
    train_yolo(pre_trained_model_path, data_to_be_trained_path, epochs)


@app.command()
def test(
    path_to_best_weights: str = typer.Option(
        "--best_weight_path", help="Path to weights generated after training"
    ),
    path_to_test_data: str = typer.Option(
        "--test_data_path", help="Path to test dataset to the test the performance"
    ),
):
    test_yolo(path_to_best_weights, path_to_test_data)


if __name__ == "__main__":
    app()
