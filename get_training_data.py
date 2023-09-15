from get_image import get_image
import keyboard
from time import sleep
from datetime import datetime
from roboflow import Roboflow

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key="dhaROl0WFgzpP000tC8o")

# Specify the project for upload
project = rf.project("3d-printing-fail-detection")



while True:
    if keyboard.is_pressed(' '):
        print("Key pressed. Exiting the loop.")
        break

    image = get_image()

    if image:
        print("Image successfully fetched.")

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

        image.save(f"training_data/{formatted_time.replace(':', '-')}.jpg")

        # Upload the image to your project
        project.upload(f"training_data/{formatted_time.replace(':', '-')}.jpg")

    sleep(.2)
