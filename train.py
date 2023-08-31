from ultralytics import YOLO

# '/home/sumbalkhan12/Test/barcode-detecion/yolov8n.pt'
# /home/sumbalkhan12/Test/barcode-detecion/large_dataset/custom.yaml
print('start')
model = YOLO('/home/sumbalkhan12/Test/barcode-detecion/runs/detect/train5/weights/best.pt')
print('Successfull')
model.train(data="/home/sumbalkhan12/Test/barcode-detecion/large_dataset/custom.yaml", epochs=2)
print('Successfull2') 
model.val()
# model=YOLO("/home/sumbalkhan12/Test/barcode-detecion/runs/detect/train18/weights/best.pt")
# model.predict(source="/home/sumbalkhan12/Test/barcode-detecion/large_dataset/yolo_test/images" ,save=True)