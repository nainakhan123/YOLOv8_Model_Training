# YOLOv8_Model_Training_barcode_detection
Barcode-detection
This project aims to develop a deep learning model able to detect a barcode in a given image. The model behind it is new version of yolo that is YOLOv8 introduced by ultralytics.

Description
Here are the 4 steps for this project :

1. Implement YOLO version 8 from ultralystics, that is used for object detection.
2. Next create an instance of yolo and load pre-trained model of yolov8.
3. Call the train method of the YOLO model instance. This initiates the training process using a custom dataset defined in the provided YAML file.
4. After training call the validate method of the model.
5. When the model is trained, run te test.py to test the model on test data.
