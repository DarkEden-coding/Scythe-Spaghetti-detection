from web_interaction.utils import get
from settings import printer_url


def get_position():
    return get(printer_url + "printer/objects/query?motion_report").json()["result"]["status"]["motion_report"]["live_position"]
