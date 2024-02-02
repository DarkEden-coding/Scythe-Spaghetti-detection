from detect_spaghetti import detect
from PIL import Image

image = Image.open("fail_images/fail_img(1).jpg")

detect(image, 0.2)
