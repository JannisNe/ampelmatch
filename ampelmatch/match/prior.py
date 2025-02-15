import abc
import logging
from typing import Literal, Union

import healpy as hp
import ipdb
import numpy as np
import pandas as pd
from cachier import cachier
from pydantic import (
    BaseModel,
    field_validator,
    PositiveInt,
)

from ampelmatch.cache import cache_dir

logger = logging.getLogger(__name__)


class BasePrior(BaseModel, abc.ABC):
    name: str

    @abc.abstractmethod
    def evaluate_prior(
        self, data: pd.DataFrame, match_data: list[pd.DataFrame]
    ) -> float: ...


class SurfaceDensityPrior(BasePrior):
    name: Literal["surface_density"]
    nside: PositiveInt
    area_sqdg: float

    @field_validator("nside")
    def check_nside(cls, v):
        if not hp.isnsideok(v):
            raise ValueError(f"nside {v} is not valid")
        return v

    @staticmethod
    @cachier(cache_dir=cache_dir)
    def compute_prior(data: list[pd.DataFrame], nside, area_sqdg) -> pd.DataFrame:
        logger.info("computing prior")
        densities = pd.DataFrame(
            index=range(hp.nside2npix(nside)),
            dtype=float,
            columns=range(len(data)),
        )
        for i, i_data in enumerate(data):
            ids_list = hp.ang2pix(
                nside=nside, theta=i_data.ra, phi=i_data.dec, lonlat=True
            )
            ids, count = np.unique(ids_list, return_counts=True)
            densities.loc[ids, i] = count / hp.nside2pixarea(nside, degrees=True)
        ipdb.set_trace()
        return densities.product(axis=1, skipna=False) / (area_sqdg ** len(data))

    def evaluate_prior(
        self, data: pd.DataFrame, match_data: list[pd.DataFrame]
    ) -> float:
        prior_per_hp_index = self.compute_prior(match_data, self.nside, self.area_sqdg)
        median_prior = prior_per_hp_index.median()
        logger.info(f"median prior: {median_prior}")
        data_hp_index = hp.ang2pix(
            nside=self.nside, theta=data.ra, phi=data.dec, lonlat=True
        )
        return prior_per_hp_index.loc[data_hp_index]


Prior = Union[SurfaceDensityPrior]
