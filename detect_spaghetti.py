import os.path
from ultralytics import YOLO
import cv2
import numpy as np
from time import time

try:
    from settings import use_cuda
except ImportError:
    print(
        "\033[91mSettings file not found. Using template settings. !!!MAKE SURE IT IS UPDATED!!!\033[0m"
    )
    from template_settings import use_cuda

device = 0 if use_cuda else "cpu"

model = YOLO("largeModel.pt")

if not os.path.exists("/fail_images"):
    os.mkdir("/fail_images")


def detect(image, min_conf):
    start_time = time()

    # convert image to grayscale
    image = image.convert("L")

    results = model(
        source=image,
        save=True,
        save_conf=True,
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

            print(f"Confidence: {conf}, Class: {class_num}\n")

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

    filename = "fail_img.jpg"
    counter = 0
    while os.path.exists(f"fail_images/{filename}"):
        counter += 1
        filename = f"fail_img({counter}).jpg"

    if len(box_list) > 0:
        cv2.imwrite(f"fail_images/{filename}", cv2_image)
        print(f"Detection took {round(time() - start_time, 2)} seconds.")
        return box_list

    print(f"Detection took {round(time() - start_time, 2)} seconds.")
    return False
