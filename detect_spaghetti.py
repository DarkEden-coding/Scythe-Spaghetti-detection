import os.path

from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/spaghetti-detection-L7/weights/best.pt")


def detect(image, min_conf):
    # convert image to grayscale

    image = image.convert("L")

    results = model(source=image, save=True, save_conf=True, show=False, conf=min_conf)

    box_list = []

    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = round(box.conf[0].item(), 2)
            if conf < min_conf:
                continue
            x1, y1, x2, y2 = int(box.xyxy[0][0].item()), int(box.xyxy[0][1].item()), int(box.xyxy[0][2].item()), int(box.xyxy[0][3].item())
            box_list.append((x1, y1, x2, y2, conf))

            cv2.rectangle(cv2_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv2_image, str(conf), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite("fail_img.jpg", cv2_image)

    filename = "fail_img.jpg"
    counter = 0
    while os.path.exists(f"/fail_images/{filename}"):
        counter += 1
        filename = f"fail_img {counter}.jpg"

    cv2.imwrite(f"/fail_images/{filename}", cv2_image)

    if len(box_list) > 0:
        return box_list
    return False
