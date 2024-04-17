from web_interaction.get_image import get_image
from web_interaction.is_printing import is_printing
from web_interaction.move_to_pos import move_to_pos
from web_interaction.wait_for_position import wait_for_position
from scythe_logging import log
from settings import bed_size as _bed_size
import os


if not os.path.exists("calibration_images"):
    os.makedirs("calibration_images")


def calibration(bed_size=_bed_size):
    """
    Calibrate the detection for your printer, this will move the printer to all positions in a grid and take/save images for training,
    do not have a print on the bed while running this function
    """
    if is_printing():
        log("Printer is currently printing, cannot calibrate while printing")
        return

    # make grid of 10x10
    x_grid = 6
    y_grid = 6

    # calculate the step size for the grid
    x_step = bed_size[0] / x_grid
    y_step = bed_size[1] / y_grid

    # move to each position in the grid and take an image
    for x in range(x_grid):
        for y in range(y_grid):
            print("Moving...")
            move_to_pos(x * x_step, y * y_step, 30)
            wait_for_position((x * x_step, y * y_step, 30))
            print("Getting image...")
            image = get_image()
            print("Saving image...")
            image.save(f"calibration_images/{x}_{y}.png")
            log(f"Saved image {x}_{y}.png")
            print(f"Percentage complete: {round(((x * y_grid) + y) / (x_grid * y_grid) * 100)}%\n")

    print("Calibration complete. You can now train the model with the images in the calibration_images folder.")


if __name__ == "__main__":
    calibration()
