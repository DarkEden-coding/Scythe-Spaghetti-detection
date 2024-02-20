from settings import printer_url
from web_interaction.utils import get


def is_printing():
    """
    Check if the printer is currently printing
    :return: bool
    """

    response = get(printer_url + "printer/objects/query?print_stats")

    return (
        "printing"
        in response.json()["result"]["status"]["print_stats"]["state"].lower()
    )
