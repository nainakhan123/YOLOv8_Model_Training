from ultralytics import YOLO

model = YOLO('/home/sumbalkhan12/Test/barcode-detecion/runs/detect/train5/weights/best.pt')
model.train(data="/home/sumbalkhan12/Test/barcode-detecion/large_dataset/custom.yaml", epochs=2)
print('Successfull2') 
model.val()
