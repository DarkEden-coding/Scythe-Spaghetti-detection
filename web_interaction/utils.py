from requests import get as _get
from requests import post as _post

from requests.exceptions import ConnectionError
from time import sleep
from scythe_logging import log


def get(url: str):
    """
    Get a URL, retrying if a connection error occurs
    :param url: the URL to get
    :return:
    """

    while True:
        try:
            return _get(url)
        except ConnectionError:
            log("Connection error, retrying...")
            sleep(1)


def post(url: str):
    """
    Post to a URL, retrying if a connection error occurs
    :param url: the URL to post to
    :return:
    """

    while True:
        try:
            return _post(url)
        except ConnectionError:
            log("Connection error, retrying...")
            sleep(1)
