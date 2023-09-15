from ultralytics import YOLO
import os

# Load the model.
model = YOLO('runs/detect/spaghetti-detection-L6/weights/best.pt')

# Training.
results = model.train(
    data='3d printing fail detection.v3i.yolov8/data.yaml',
    imgsz=640,
    epochs=80,
    batch=-1,
    name='spaghetti-detection-L7',
    cache=True,
)
