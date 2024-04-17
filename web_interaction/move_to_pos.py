from settings import printer_url
from web_interaction.utils import post
from scythe_logging import log


def move_to_pos(x, y, z):
    """
    Move the printer to a specific position
    :param x: the x coordinate to move to
    :param y: the y coordinate to move to
    :param z: the z coordinate to move to
    :return:
    """
    post(printer_url + f"printer/gcode/script?script=G0 X{x} Y{y} Z{z} F10000")
    log(f"Moving to position ({x}, {y}, {z})")
