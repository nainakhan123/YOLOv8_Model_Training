from pydantic import BaseModel

class TrainRequest(BaseModel):
    pre_trained_model_path: str
    dataset_config_path: str
    epochs: int


class TestRequest(BaseModel):
    path_to_best_weights: str
    path_to_test_data: str