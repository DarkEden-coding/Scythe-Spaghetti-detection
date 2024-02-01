from ultralytics import YOLO


def main():
    # Load the model.
    model = YOLO(
        "runs/detect/spaghetti-detection-L11/weights/best.pt"
    )  # change to last trained model

    # Training.
    model.train(
        data="3d printing fail detection.v4i.yolov8/data.yaml",  # change dataset path
        imgsz=640,
        epochs=360,
        batch=-1,
        name="spaghetti-detection-L12",  # change to new model name
        cache=True,
        device=0,  # 0 is the first GPU, remove this line to use CPU, use array to use multiple GPUs
    )


if __name__ == "__main__":
    main()
