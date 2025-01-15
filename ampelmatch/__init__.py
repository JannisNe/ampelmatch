import logging
from rich.logging import RichHandler

formatter = logging.Formatter('%(message)s', "%H:%M:%S")
handler = RichHandler()
handler.setFormatter(formatter)
logging.getLogger("ampelmatch").addHandler(handler)
