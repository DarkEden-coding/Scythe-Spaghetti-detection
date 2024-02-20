from requests import get as _get
from requests import post as _post

from requests.exceptions import ConnectionError
from time import sleep


def get(url):
    while True:
        try:
            return _get(url)
        except ConnectionError:
            print("Connection error, retrying...")
            sleep(1)


def post(url):
    while True:
        try:
            return _post(url)
        except ConnectionError:
            print("Connection error, retrying...")
            sleep(1)
