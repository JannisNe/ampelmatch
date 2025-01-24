import logging
from ampelmatch.data.config import DatasetConfig
from ampelmatch.data.plotter import Plotter


logger = logging.getLogger("ampelmatch.data.make_plots")


fields = {
    0: {'ra': +150.11916667, 'dec': +2.20583333},
    1: {'ra': +10.11916667, 'dec': -2.60583333}
}
survey_params = [
    {
        "name": "survey1",
        "survey_type": "PositionalGridSurvey",
        "fields": fields,
        'fov': 10,
        'gain': 1,
        'zp': 30,
        'skynoise_mean': 150,
        'time_min': "2020-03-01",
        'time_max': "2021-02-28",
        'bands': ['ztfr'],
        "size": 100,
        "uncertainty": {
            "type": "GaussianUncertainty",
            "sigma_arcsec": 1
        },
    },
    {
        "name": "survey2",
        "survey_type": "PositionalGridSurvey",
        "fields": {k: {'ra': v['ra'] + 0.1, 'dec': v['dec'] - 0.2} for k, v in fields.items()},
        'fov': 5,
        'gain': 1,
        'zp': 28,
        'skynoise_mean': 200,
        'time_min': "2020-03-01",
        'time_max': "2021-02-28",
        'bands': ['atlaso'],
        'size': 100,
        "uncertainty": {
            "type": "GaussianUncertainty",
            "sigma_arcsec": 3
        }
    }
]
transient_config = [
    {
        "name": "SNIa",
        "draw": 10_000,
        "zmax": 1,
        "tstart": "2020-01-01",
        "tstop": "2021-03-15"
    }
]
dataset_config = {
    "name": "test",
    "surveys": survey_params,
    "transients": transient_config
}


if __name__ == '__main__':
    logging.getLogger("ampelmatch").setLevel("DEBUG")
    logging.getLogger("ampelmatch").info("Generating test data")
    c = DatasetConfig.model_validate(dataset_config)
    Plotter(c).make_plots()
