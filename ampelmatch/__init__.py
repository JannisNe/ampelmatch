import logging
from rich.logging import RichHandler

cache_dir = "ampelmatch_cache"

formatter = logging.Formatter('%(message)s', "%H:%M:%S")
handler = RichHandler()
handler.setFormatter(formatter)
logging.getLogger("ampelmatch").addHandler(handler)
