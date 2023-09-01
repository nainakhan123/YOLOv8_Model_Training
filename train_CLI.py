import argparse
from ultralytics import YOLO

def train_yolo(weights_path, data_config, epochs):
    print('Starting training...')
    model = YOLO(weights_path)
    model.train(data=data_config, epochs=epochs)
    print('Training completed.')

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on custom data')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data-config', required=True)
    parser.add_argument('--epochs', type=int, default=2)
    
    args = parser.parse_args()
    
    train_yolo(args.weights, args.data_config, args.epochs)

if __name__ == '__main__':
    main()
