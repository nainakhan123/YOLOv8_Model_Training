from ultralytics import YOLO
import argparse


def predict_yolo(path_to_weights, path_to_test):
    model = YOLO(path_to_weights)
    model.predict(source=path_to_test)


def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument('path_to_weights')
    parser.add_argument('path_to_test')
    

    args = parser.parse_args()


    predict_yolo(args.path_to_weights, args.path_to_test)

if __name__ == '__main__':
    main()
