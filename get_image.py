import time
import requests
from PIL import Image
from io import BytesIO

# URL of the image
image_url = "http://mainsailos.local//webcam/?action=snapshot"


def get_image():
    # Fetch the image from the URL
    try:
        response = requests.get(image_url)
    except:
        return False

    # Check if the request was successful
    if response.status_code == 200:
        # Convert the image content to a PIL Image object
        image_data = BytesIO(response.content)
        img = Image.open(image_data)

        return img
    else:
        return False
