from requests import get as _get
from requests import post as _post

from requests.exceptions import ConnectionError, HTTPError
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
            response = _get(url)
            response.raise_for_status()
            return response
        except (ConnectionError, HTTPError):
            log("Connection error, retrying...")
            sleep(0.5)


def post(url: str):
    """
    Post to a URL, retrying if a connection error occurs
    :param url: the URL to post to
    :return:
    """

    while True:
        try:
            response = _post(url)
            response.raise_for_status()
            return response
        except (ConnectionError, HTTPError):
            log("Connection error, retrying...")
            sleep(0.5)
