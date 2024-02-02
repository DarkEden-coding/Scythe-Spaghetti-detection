import requests

url = "http://mainsailos.local/"


def is_printing():
    response = requests.get(url + "printer/objects/query?print_stats")

    return (
        "printing"
        in response.json()["result"]["status"]["print_stats"]["state"].lower()
    )
