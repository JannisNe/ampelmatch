import logging
import skysurvey
import pandas as pd
import numpy as np
from shapely import geometry
from astropy.time import Time
import hashlib
from pathlib import Path

from ampelmatch.data.positional_uncertainty import GaussianUncertainty
from ampelmatch.data.positional_survey import PositionalGridSurvey
from ampelmatch.data.positional_dataset import PositionalDataset


logger = logging.getLogger("ampelmatch.data.generate_test_data")


def generate_transient_sample():
    snia = skysurvey.SNeIa().draw(10_000)
    snia.to_csv("snia.csv")

field1 = {'ra': +150.11916667, 'dec': +2.20583333}
survey_params = {
    "survey1": {
        "fields": {0: field1},
        'fov': 8,
        'gain': 1,
        'zp': 30,
        'skynoise': 150,
        'time_min': "2020-03-01",
        'time_max': "2020-04-01",
        'bands': ['ztfg'],
        "size": 10_000
    },
    "survey2": {
        "fields": {0: {'ra': field1["ra"] + 2, 'dec': field1["dec"] + 0.4}},
        'fov': 5,
        'gain': 1,
        'zp': 30,
        'skynoise_mean': 150,
        'time_min': "2020-03-01",
        'time_max': "2020-04-01",
        'bands': ['ztfg'],
        'size': 10_000
    }
}
_one_degree_vertices = np.asarray([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])

def get_test_observations(survey_name: str):
    """
    generate observations by two overlapping mock surveys that observe one field each
    :return:
    """

    params = survey_params[survey_name]
    h = hashlib.md5().hexdigest(params)
    fname = Path(f"{survey_name}_{h}.csv")
    if not fname.exists():
        logger.info(f"generating {survey_name} observations")
        size = params["size"]
        data = {}
        data["fieldid"] = np.random.choice(list(params["fields"].keys()), size=size)
        data["gain"] = params["gain"]
        data["zp"] = params["zp"]
        data["skynoise"] = np.random.normal(loc=params["skynoise_mean"], scale=20, size=size)
        data["mjd"] = np.random.uniform(Time(params["time_min"]).mjd, Time(params["time_max"]).mjd, size=size)
        data["band"] = np.random.choice(params["bands"], size=size)
        data = pandas.DataFrame.from_dict(data)
        logger.info(f"generated {size} observations for survey {survey_name}")
        data.to_csv(fname)
        logger.info(f"saved to {fname}")
    return pd.read_csv(fname, index_col=0)


def generate_test_surveys():
    for survey_name in survey_params:
        survey = PositionalGridSurvey()


if __name__ == "__main__":
    logging.getLogger("ampelmatch").setLevel(logging.DEBUG)
    # generate_test_data()
    ge()
