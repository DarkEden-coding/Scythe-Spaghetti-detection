import requests
from PIL import Image
from io import BytesIO
from settings import printer_url, webcam_name

webcam_list = requests.get(printer_url + "server/webcams/list").json()["result"][
    "webcams"
]

snapshot_url = None

# Find the webcam URL
for webcam in webcam_list:
    if webcam["name"] == webcam_name:
        snapshot_url = webcam["snapshot_url"]
        break
if not snapshot_url:
    raise ValueError(f"Webcam '{webcam_name}' not found, the name is case sensitive")


def get_image():
    # Fetch the image from the URL
    try:
        response = requests.get(printer_url + snapshot_url)
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
