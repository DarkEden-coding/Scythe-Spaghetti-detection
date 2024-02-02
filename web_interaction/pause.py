import requests

url = "http://mainsailos.local/"


def pause():
    requests.post(url + "printer/print/pause")


pause()
