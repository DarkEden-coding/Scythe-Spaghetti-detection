from ultralytics import YOLO


def util_export(path):
    # Load a model
    print("Loading model...")
    model = YOLO(path)
    print("Model loaded.")

    # Export the model
    print("Exporting model...")
    model.export(
        format="onnx", imgsz=640, device="cpu"
    )  # settings for exporting to co processer
    print("Model exported.")


if __name__ == "__main__":
    util_export(
        input("Enter the path of the model that you want to export (ending in .pt): ")
    )
