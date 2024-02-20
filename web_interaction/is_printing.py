from settings import printer_url
from utils import get


def is_printing():
    response = get(printer_url + "printer/objects/query?print_stats")

    return (
        "printing"
        in response.json()["result"]["status"]["print_stats"]["state"].lower()
    )
