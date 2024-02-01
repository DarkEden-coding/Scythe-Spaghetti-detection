from ultralytics import YOLO


def main():
    # Load the model.
    model = YOLO('runs/detect/spaghetti-detection-L11/weights/best.pt')

    # Training.
    model.train(
        data='3d printing fail detection.v4i.yolov8/data.yaml',
        imgsz=640,
        epochs=360,
        batch=-1,
        name='spaghetti-detection-L12',
        cache=True,
        device=0,
    )


if __name__ == '__main__':
    main()
