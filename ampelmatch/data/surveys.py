import logging
import pandas as pd
import numpy as np
from shapely import geometry
from astropy.time import Time
import hashlib
import json
from pathlib import Path

from ampelmatch.data.positional_uncertainty import BaseUncertainty
from ampelmatch.data.positional_survey import PositionalGridSurvey


logger = logging.getLogger(__name__)


fields = {
    0: {'ra': +150.11916667, 'dec': +2.20583333},
    1: {'ra': +10.11916667, 'dec': -2.60583333}
}
survey_params = {
    "survey1": {
        "fields": fields,
        'fov': 8,
        'gain': 1,
        'zp': 30,
        'skynoise_mean': 150,
        'time_min': "2020-03-01",
        'time_max': "2020-04-01",
        'bands': ['ztfg'],
        "size": 10_000,
        "uncertainty": {
            "type": "GaussianUncertainty",
            "sigma_arcsec": 1
        }
    },
    "survey2": {
        "fields": {k: {'ra': v['ra'] + 0.1, 'dec': v['dec'] - 0.2} for k, v in fields.items()},
        'fov': 5,
        'gain': 1,
        'zp': 30,
        'skynoise_mean': 150,
        'time_min': "2020-03-01",
        'time_max': "2020-04-01",
        'bands': ['ztfg'],
        'size': 10_000,
        "uncertainty": {
            "type": "GaussianUncertainty",
            "sigma_arcsec": 3
        }
    }
}


def get_survey_hash(survey_name: str):
    params = survey_params[survey_name]
    return hashlib.md5(json.dumps(params).encode()).hexdigest()


def get_test_observations(survey_name: str):
    """
    generate observations by two overlapping mock surveys that observe one field each
    :return:
    """

    params = survey_params[survey_name]
    h = get_survey_hash(survey_name)

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
        data = pd.DataFrame.from_dict(data)
        logger.info(f"generated {size} observations for survey {survey_name}")
        data.to_csv(fname)
        logger.info(f"saved to {fname}")
    return pd.read_csv(fname, index_col=0)


def generate_test_surveys():
    for survey_name, p in survey_params.items():
        logger.debug(f"generating survey {survey_name}: {p}")
        uncertainty = BaseUncertainty.from_dict(dict(p["uncertainty"]))
        data = get_test_observations(survey_name)
        _one_degree_vertices = np.asarray([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        footprint = geometry.Polygon(_one_degree_vertices * p["fov"])
        yield survey_name, PositionalGridSurvey.from_pointings(data, p["fields"], footprint, uncertainty=uncertainty)
