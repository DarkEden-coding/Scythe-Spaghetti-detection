from ultralytics import YOLO
from roboflow import Roboflow
from settings import roboflow_api_key

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key=roboflow_api_key)

# Specify the project for upload
project = rf.project("3d-printing-fail-detection")
version = project.version("6")


def main():
    print("Loading model...")

    # Load the model.
    model = YOLO("yolov8s.pt")  # change to last trained model

    print("Model loaded.")
    print("Starting training...")

    # Training.
    model.train(
        data="3d printing fail detection.v6i.yolov8/data.yaml",  # change dataset path
        imgsz=640,
        epochs=180,
        batch=-1,
        name="spaghetti-detection-S1",  # change to new model name
        cache=True,
        device=0,  # 0 is the first GPU, remove this line to use CPU, use array to use multiple GPUs
    )

    print("Training complete.")
    print("Uploading model...")

    # Upload the model to your project
    version.deploy(
        "yolov8",
        "runs/detect/spaghetti-detection-S1/",
    )

    print("Model uploaded.")


if __name__ == "__main__":
    main()
