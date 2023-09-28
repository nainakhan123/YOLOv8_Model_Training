# YOLOv8_Model_Training_barcode_detection
Barcode-detection
This project aims to develop a deep learning model able to detect a barcode in a given image. The model behind it is new version of yolo that is YOLOv8 introduced by ultralytics.

Description
Here are the 4 steps for this project :

1. Implement YOLO version 8 from ultralystics, that is used for object detection.
2. Next create an instance of yolo and load pre-trained model of yolov8.
3. Now through CLI run the command 
```bash 
python cli.py --pre_trained_model_path model.pt --dataset_to_be_trained_config_path  --epochs 2
```,
 replace the placeholders path_to_trained_model and path_to_data_to_train with your actual paths and train the model. 
4. After training the model, test the performance on test dataset running the the command through CLI
 ```bash
 python cli.py --path_to_best weights weights.pt --path_to_test_data test_data
 ```. 
 Replace the placeholder with your dataset path.
 5. For using the api functionality of our application you can use it by simply running the following command on the command prompt 
 ```bash
  uvicorn api:app --reload
 ```
6. After opening the application in browser in the url go to api docs and from their you can train and test the dataset using YOLOv8.

Installation

1. Install python version 3.11.3.
2. Setup the virtual environment
3. Install the requirements mentioned in the requirements.txt by the running the command 
```bash
pip install -r requirements.txt
```
4. Clone this repository.
