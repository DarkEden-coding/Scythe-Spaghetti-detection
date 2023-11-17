import time
import requests
from PIL import Image
from io import BytesIO

# URL of the image
image_url = "http://mainsailos.local//webcam/?action=snapshot"
ip_image_url = "http://10.0.0.212//webcam/?action=snapshot"
image_url = ip_image_url


def get_image():
    # Fetch the image from the URL
    try:
        response = requests.get(image_url)
    except:
        try:
            response = requests.get(ip_image_url)
        except:
            return False

    # Check if the request was successful
    if response.status_code == 200:
        # Convert the image content to a PIL Image object
        image_data = BytesIO(response.content)
        img = Image.open(image_data)

        # save the image to current_view.jpg
        img.save("current_view.jpg")

        return img
    else:
        response = requests.get(ip_image_url)
        # Check if the request was successful
        if response.status_code == 200:
            # Convert the image content to a PIL Image object
            image_data = BytesIO(response.content)
            img = Image.open(image_data)

            # save the image to current_view.jpg
            img.save("current_view.jpg")

            return img
        else:
            return False
