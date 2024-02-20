import os.path
from ultralytics import YOLO
import cv2
import numpy as np
from time import time
from model_utils.onnx_export import util_export
from logging import log

if not os.path.exists("settings.py"):
    log("settings.py not found. Please run settings_ui.py first.")
    exit()

from settings import use_cuda, use_onnx


device = 0 if use_cuda else "cpu"

if use_onnx:
    # if onnx model is there then use it, otherwise run onnx_export.py
    if not os.path.exists("model_utils/largeModel.onnx"):
        log("No ONNX model found. Building model...")
        util_export("model_utils/largeModel.pt")
        log("Model built.")

log("Loading YOLO model...")

model_path = "model_utils/largeModel.onnx" if use_onnx else "model_utils/largeModel.pt"

model = YOLO(model_path, task="detect")

log("YOLO model loaded.")


def detect(image, min_conf):
    start_time = time()

    # convert image to grayscale
    image = image.convert("L")

    # resize image to 640x640
    image = image.resize((640, 640))

    results = model(
        source=image,
        save=False,
        save_conf=False,
        show=False,
        conf=min_conf,
        device=device,
    )

    box_list = []

    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = round(box.conf[0].item(), 2)
            class_num = box.cls[0].item()

            log(f"Confidence: {conf}, Class: {class_num}\n")

            if class_num != 1:
                continue

            if conf < min_conf:
                continue

            x1, y1, x2, y2 = (
                int(box.xyxy[0][0].item()),
                int(box.xyxy[0][1].item()),
                int(box.xyxy[0][2].item()),
                int(box.xyxy[0][3].item()),
            )
            box_list.append((x1, y1, x2, y2, conf))

            cv2.rectangle(cv2_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                cv2_image,
                str(conf),
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    cv2.imwrite("fail_img.jpg", cv2_image)

    if len(box_list) > 0:
        log(f"Detection took {round(time() - start_time, 2)} seconds.")
        return box_list

    log(f"Detection took {round(time() - start_time, 2)} seconds.")
    return False
