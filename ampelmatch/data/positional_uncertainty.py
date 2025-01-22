import logging
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


logger = logging.getLogger(__name__)


class BaseUncertainty:
    registry = {}
    POSITION_KEYS = ["ra", "dec"]
    PARAMETER_KEYS = []

    def draw_position(self, lc_in: pd.DataFrame, truth: pd.Series) -> tuple:
        ...

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ in BaseUncertainty.registry:
            raise ValueError(f"Duplicate uncertainty class name: {cls.__name__}")
        BaseUncertainty.registry[cls.__name__] = cls

    @classmethod
    def from_dict(cls, data: dict) -> 'BaseUncertainty':
        logger.debug(f"Creating uncertainty from dict: {data}")
        if "type" not in data:
            raise ValueError("Uncertainty type not provided")
        _type = data.pop("type")
        if _type not in BaseUncertainty.registry:
            raise ValueError(f"Uncertainty type not recognized: {_type}")
        return BaseUncertainty.registry[_type](**data)


class GaussianUncertainty(BaseUncertainty):

    PARAMETER_KEYS = ["sigma_arcsec"]

    def __init__(self, sigma_arcsec: float):
        self.sigma = sigma_arcsec

    def draw_position(self, lc_in: pd.DataFrame, truth: pd.Series) -> tuple:
        coords = SkyCoord(truth["ra"], truth["dec"], unit='deg')
        offsets = np.random.normal(0, self.sigma, len(lc_in))
        ps = np.random.uniform(0, 2 * np.pi, len(lc_in))
        new_coords = coords.directional_offset_by(ps, offsets * u.arcsec)
        return new_coords.ra.deg, new_coords.dec.deg, [self.sigma] * len(lc_in)
