import typer
from ultralytics import YOLO


app = typer.Typer()


@app.command()
def test_yolo(
    path_to_best_weights: str = typer.Argument(),
    path_to_test_file: str = typer.Argument(),
):
    print("Testing started")
    model: YOLO = YOLO(path_to_best_weights)
    model.predict(path_to_test_file, save=True)
    print("Testing done successfully")


if __name__ == "__main__":
    app()
