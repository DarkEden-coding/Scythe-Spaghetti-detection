from detect_spaghetti import detect
from PIL import Image

image = Image.open("fail_img1.jpg")

detect(image, 0.2)
