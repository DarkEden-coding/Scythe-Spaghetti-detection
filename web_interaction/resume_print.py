from settings import printer_url
from web_interaction.is_printing import is_printing
from web_interaction.utils import post


def resume():
    """
    Resume the print
    :return:
    """

    if not is_printing():
        post(printer_url + "printer/print/resume")
