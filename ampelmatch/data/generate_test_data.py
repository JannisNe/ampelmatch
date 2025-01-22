import logging
import skysurvey
import pandas as pd
import numpy as np
from shapely import geometry
from astropy.time import Time

from ampelmatch.data.positional_uncertainty import GaussianUncertainty
from ampelmatch.data.positional_survey import PositionalGridSurvey
from ampelmatch.data.positional_dataset import PositionalDataset


logger = logging.getLogger("ampelmatch.data.generate_test_data")


def generate_transient_sample():
    snia = skysurvey.SNeIa().draw(10_000)
    snia.to_csv("snia.csv")


def generate_test_observations():
    """
    generate observations by two overlapping mock surveys that observe one field each
    :return:
    """

    _one_degree_vertices = np.asarray([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])

    # survey 1
    field1 = {'ra': +150.11916667, 'dec': +2.20583333}
    vertices_camera1 = _one_degree_vertices * 8
    footprint1 = geometry.Polygon(vertices_camera1)
    size1 = 10_000
    data1 = {}
    data1["fieldid"] = np.array([0] * size1)
    data1["gain"] = 1
    data1["zp"] = 30
    data1["skynoise"] = np.random.normal(loc=150, scale=20, size=size1)
    data1["mjd"] = np.random.uniform(Time("2020-03-01").mjd, Time("2020-04-01").mjd, size=size1)
    data1["band"] = np.array(["ztfg"] * size1)
    data1 = pd.DataFrame.from_dict(data1)
    logger.info(f"Generated {size1} observations for survey 1")
    data1.to_csv("survey1.csv")
    logger.info(f"Saved survey 1 to survey1.csv")

    # survey 2
    field2 = {'ra': field1["ra"] + 2, 'dec': field1["dec"] + 0.4}
    vertices_camera2 = _one_degree_vertices * 5
    footprint2 = geometry.Polygon(vertices_camera2)
    size2 = 10_000
    data2 = {}
    data2["fieldid"] = np.array([0] * size2)
    data2["gain"] = 1
    data2["zp"] = 20
    data2["skynoise"] = np.random.normal(loc=150, scale=20, size=size2)
    data2["mjd"] = np.random.uniform(Time("2020-03-01").mjd, Time("2020-04-01").mjd, size=size2)
    data2["band"] = np.array(["atlaso"] * size2)
    data2 = pd.DataFrame.from_dict(data2)
    logger.info(f"Generated {size2} observations for survey 2")
    data2.to_csv("survey2.csv")
    logger.info(f"Saved survey 2 to survey2.csv")


if __name__ == "__main__":
    logging.getLogger("ampelmatch").setLevel(logging.DEBUG)
    # generate_test_data()
    generate_test_observations()
