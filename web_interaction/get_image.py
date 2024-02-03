import requests
from PIL import Image
from io import BytesIO
from settings import printer_url


def get_image():
    # Fetch the image from the URL
    try:
        response = requests.get(printer_url)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
        return False

    # Check if the request was successful
    if response.status_code == 200:
        # Convert the image content to a PIL Image object
        image_data = BytesIO(response.content)
        img = Image.open(image_data)

        return img
    else:
        return False
