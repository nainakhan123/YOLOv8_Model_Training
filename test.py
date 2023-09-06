from ultralytics import YOLO

# /home/sumbalkhan12/Test/barcode-detecion/large_dataset/yolo_test/images
model = YOLO("/home/sumbalkhan12/Test/barcode-detecion/runs/detect/train5/weights/best.pt")
model.predict(source="test4.png", save=True)