from web_interaction.get_position import get_position
from time import sleep


def wait_for_position(position, tolerance=0.1):
    while True:
        current_position = get_position()[:-1]
        if abs(current_position[0] - position[0]) < tolerance and abs(current_position[1] - position[1]) < tolerance:
            break
        sleep(0.1)
