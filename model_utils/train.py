from ultralytics import YOLO


def main():
    print("Loading model...")

    # Load the model.
    model = YOLO("largeModel.pt")  # change to last trained model

    print("Model loaded.")
    print("Starting training...")

    # Training.
    model.train(
        data="O:\Python Files\Projects\print-fail-detection\personal_printer_augmentation.v1i.yolov8\data.yaml",  # change dataset path
        imgsz=640,
        epochs=50,
        batch=-1,
        name="modif-spaghetti-detection-L15",  # change to new model name
        cache=True,
        device=0,  # 0 is the first GPU, remove this line to use CPU, use array to use multiple GPUs
    )

    print("Training complete.")


if __name__ == "__main__":
    main()
