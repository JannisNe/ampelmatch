import logging
import skysurvey


logger = logging.getLogger("ampelmatch.data.generate_test_data")


def generate_test_data():
    snia = skysurvey.SNeIa().draw(10_000)
    snia.to_csv("snia.csv")


if __name__ == "__main__":
    generate_test_data()