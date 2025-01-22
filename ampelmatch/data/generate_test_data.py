import logging
import skysurvey
import pandas as pd
import ampelmatch


logger = logging.getLogger("ampelmatch.data.generate_test_data")


def generate_transient_sample():
    snia = skysurvey.SNeIa().draw(10_000)
    snia.to_csv("snia.csv")


def generate_test_data():
    survey1 = skysurvey.ZTF.from_logs()
    survey2 = skysurvey.DES.from_logs()
    targets_data = pd.read_csv("snia.csv")
    logger.info(f"Loaded {len(targets_data)} targets")
    targets = skysurvey.SNeIa()
    targets.set_data(targets_data)

    logger.info("Creating observations")
    obs1 = skysurvey.DataSet.from_targets_and_survey(targets, survey1)
    obs2 = skysurvey.DataSet.from_targets_and_survey(targets, survey2)


if __name__ == "__main__":
    logging.getLogger("ampelmatch").setLevel(logging.DEBUG)
    generate_test_data()
