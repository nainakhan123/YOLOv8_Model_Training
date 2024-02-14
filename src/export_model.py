from ultralytics import YOLO


model = YOLO('yolov8n.pt') 
model = YOLO('/home/sumbalkhan12/YOLOv8_Model_Training/src/best.pt') 

model.export(format='coreml')
