import logging
import skysurvey
import numpy as np
import pandas as pd
from astropy.time import Time
import diskcache
from shapely import geometry
from abc import ABC, abstractmethod

from ampelmatch import cache_dir
from ampelmatch.data.positional_uncertainty import BaseUncertainty
from ampelmatch.data.config import PositionalGridSurveyConfig, BasePositionalSurveyConfig


logger = logging.getLogger(__name__)


cache = diskcache.Cache(cache_dir)


class ObservationRealize(ABC):

    @classmethod
    @abstractmethod
    def from_config(cls, config: BasePositionalSurveyConfig):
        ...

    @classmethod
    @abstractmethod
    def realize_observations(cls, config: BasePositionalSurveyConfig):
        ...


class PositionalGridSurvey(skysurvey.GridSurvey, ObservationRealize):

    def __init__(self, uncertainty: BaseUncertainty, data=None, fields=None):
        super().__init__(data, fields)
        self.uncertainty = uncertainty

    @classmethod
    def from_pointings(cls, data, fields_or_coords=None, footprint=None, uncertainty: BaseUncertainty = None, **kwargs):
        _survey = super(cls, cls).from_pointings(data, fields_or_coords, footprint, **kwargs)
        _survey.uncertainty = uncertainty
        return _survey

    @classmethod
    def from_config(cls, config: PositionalGridSurveyConfig):
        data = cls.realize_observations(config)
        uncertainty = BaseUncertainty.from_dict(config.uncertainty.model_dump())
        _one_degree_vertices = np.asarray([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        footprint = geometry.Polygon(_one_degree_vertices * config.fov)
        return cls.from_pointings(data, fields=config.fields, uncertainty=uncertainty, footprint=footprint)

    @classmethod
    @cache.memoize()
    def realize_observations(cls, config: PositionalGridSurveyConfig):
        logger.info(f"generating {config.name} observations")
        data = {
            "fieldid": np.random.choice(list(config.fields.keys()), size=config.size),
            "gain": config.gain,
            "zp": config.zp,
            "skynoise": np.random.normal(loc=config.skynoise_mean, scale=20, size=config.size),
            "mjd": np.random.uniform(Time(config.time_min).mjd, Time(config.time_max).mjd, size=config.size),
            "band": np.random.choice(config.bands, size=config.size)
        }
        data = pd.DataFrame.from_dict(data)
        logger.info(f"generated {config.size} observations for survey {config.name}")
        return data
