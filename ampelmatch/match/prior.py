import abc
import logging
from typing import Literal, Union

import healpy as hp
import numpy as np
import pandas as pd
from ampelmatch.cache import cache_dir, compute_density_hash
from cachier import cachier
from pydantic import (
    BaseModel,
    field_validator,
    PositiveInt,
)

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
    cache: dict = {}

    @field_validator("nside")
    def check_nside(cls, v):
        if not hp.isnsideok(v):
            raise ValueError(f"nside {v} is not valid")
        return v

    @staticmethod
    @cachier(cache_dir=cache_dir, hash_func=compute_density_hash)
    def compute_densities(data: tuple[pd.DataFrame], nside) -> pd.DataFrame:
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
            densities.loc[ids, i] = count / hp.nside2pixarea(nside)
        logger.info(f"median density per sr \n{densities.median().to_string()}")
        return (
            densities.product(axis=1, skipna=False) * (4 * np.pi) ** len(data)
        ) ** -1

    def get_densities(self, data: tuple[pd.DataFrame], nside) -> pd.DataFrame:
        h = compute_density_hash([], {"data": data, "nside": nside})
        if h not in self.cache:
            logger.debug(f"cache miss: {h}")
            self.cache[h] = self.compute_densities(data, nside)
        return self.cache[h]

    def evaluate_prior(
        self, data: pd.DataFrame, match_data: list[pd.DataFrame]
    ) -> float:
        prior_per_hp_index = self.get_densities(match_data, self.nside)
        ra = data.ra.median()
        dec = data.dec.median()
        data_hp_index = hp.ang2pix(nside=self.nside, theta=ra, phi=dec, lonlat=True)
        return prior_per_hp_index.loc[data_hp_index]


Prior = Union[SurfaceDensityPrior]
