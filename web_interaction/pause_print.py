import requests
from settings import printer_url
from is_printing import is_printing


def pause():
    if is_printing():
        requests.post(printer_url + "printer/print/pause")
