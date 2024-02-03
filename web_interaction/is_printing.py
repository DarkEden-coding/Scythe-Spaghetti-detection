import requests
from settings import printer_url


def is_printing():
    response = requests.get(printer_url + "printer/objects/query?print_stats")

    return (
        "printing"
        in response.json()["result"]["status"]["print_stats"]["state"].lower()
    )
