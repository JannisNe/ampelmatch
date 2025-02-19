import abc
import logging
from functools import cached_property
from typing import Literal, Union

import healpy as hp
import ipdb
import numpy as np
import pandas as pd
from ampelmatch.cache import cache_dir, compute_density_hash
from cachier import cachier
from pydantic import BaseModel, field_validator, PositiveInt, computed_field, ConfigDict

logger = logging.getLogger(__name__)


class BasePrior(BaseModel, abc.ABC):
    name: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abc.abstractmethod
    def evaluate(self, data: pd.DataFrame) -> float: ...

    def __call__(self, data: pd.DataFrame) -> float:
        return self.evaluate(data)


class SurfaceDensityPrior(BasePrior, frozen=True):
    name: Literal["surface_density"]
    primary_data: dict
    match_data: list[dict]
    nside: PositiveInt
    cache: dict = {}

    @field_validator("nside")
    def check_nside(cls, v):
        if not hp.isnsideok(v):
            raise ValueError(f"nside {v} is not valid")
        return v

    @computed_field
    @cached_property
    def primary_data_df(self) -> pd.DataFrame:
        return pd.read_csv(**self.primary_data)

    @computed_field
    @cached_property
    def match_data_df(self) -> list[pd.DataFrame]:
        return [pd.read_csv(**d) for d in self.match_data]

    @computed_field
    @cached_property
    def densities(self) -> pd.DataFrame:
        return self.compute_densities(
            [self.primary_data_df] + self.match_data_df, self.nside
        )

    @staticmethod
    @cachier(cache_dir=cache_dir, hash_func=compute_density_hash)
    def compute_densities(data: tuple[pd.DataFrame], nside) -> pd.DataFrame:
        logger.info("computing prior")
        densities = pd.DataFrame(
            index=range(hp.nside2npix(nside)),
            dtype=float,
            columns=range(len(data)),
        )
        # assume that the first data entry is the primary data and group by source index
        c = ["ra", "dec"]
        pix_area = hp.nside2pixarea(nside)
        for i, i_data in enumerate(data):
            i_data = (
                i_data[c].groupby(level=0).median()
                if i == 0
                else i_data[c + ["source_index"]].groupby("source_index").median()
            )
            ids_list = hp.ang2pix(
                nside=nside, theta=i_data.ra, phi=i_data.dec, lonlat=True
            )
            ids, count = np.unique(ids_list, return_counts=True)
            densities.loc[ids, i] = count / pix_area
        logger.info(f"median density per sr \n{densities.median().to_string()}")
        min_data_length = np.argmin(len(d) for d in data)
        densities.fillna(0, inplace=True)
        p = densities[densities.columns[min_data_length]] / (
            densities.product(axis=1, skipna=False) * (4 * np.pi) ** (len(data) - 1)
        )
        logger.info(f"median prior {p.median()}")
        return p

    def evaluate(self, data: pd.DataFrame) -> float:
        ra = data.ra.median()
        dec = data.dec.median()
        data_hp_index = hp.ang2pix(nside=self.nside, theta=ra, phi=dec, lonlat=True)
        # TODO: decide whether to use interpolation here
        return self.densities.loc[data_hp_index]


Prior = Union[SurfaceDensityPrior]
