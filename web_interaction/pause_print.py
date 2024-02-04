import requests
from settings import printer_url


def pause():
    requests.post(printer_url + "printer/print/pause")
