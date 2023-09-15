from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/spaghetti-detection/weights/best.pt")

# Directory containing the images
image_dir = "3d printing fail detection.v1i.yolov8/train/images"

# Loop through each image in the directory
for image_filename in os.listdir(image_dir):
    if image_filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(image_dir, image_filename)
        im2 = cv2.imread(image_path)

        # Perform object detection
        results = model(source=im2)

        # Draw bounding boxes on the image
        for result in results:
            boxes = result.boxes

            for box in boxes:
                conf = box.conf[0]

                x1, y1, x2, y2 = int(box.xyxy[0][0].item()), int(box.xyxy[0][1].item()), int(box.xyxy[0][2].item()), int(box.xyxy[0][3].item())

                # Draw the rectangle
                cv2.rectangle(im2, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Display the confidence value near the top-left corner of the rectangle
                text = f"Conf: {conf:.2f}"
                cv2.putText(im2, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the output image with bounding boxes
        cv2.imshow("Object Detection", im2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
