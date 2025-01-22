import logging
from ampelmatch.data.dataset import generate_test_data


if __name__ == '__main__':
    logging.getLogger("ampelmatch").setLevel("INFO")
    list(generate_test_data())
