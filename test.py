import typer
from ultralytics import YOLO


app = typer.Typer()


@app.command()
def test_yolo(
    weights_path: str = typer.Argument(),
    test_path: str = typer.Argument(),
):
    print("Testing started")
    model = YOLO(weights_path)
    model.predict(test_path, save=True)
    print("Testing done successfully")


if __name__ == "__main__":
    app()
