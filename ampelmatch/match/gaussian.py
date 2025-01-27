import logging
import numpy as np
import pandas as pd
import healpy as hp
from tqdm import tqdm
from pydantic import BaseModel, computed_field, ConfigDict

import ampelmatch


logger = logging.getLogger(__name__)


class StreamMatch(BaseModel):
    nside: int
    primary_data: dict
    match_data: list[dict]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    def _primary_data(self) -> pd.DataFrame:
        logger.info(f"Loading primary data")
        return pd.read_csv(**self.primary_data)

    @computed_field
    def _match_data(self) -> list[pd.DataFrame]:
        logger.info("Loading match data")
        return [pd.read_csv(**d) for d in self.match_data]

    def match(self):
        logger.info("Matching streams")
        primary_data = self._primary_data
        match_data = self._match_data
        match_data_hp_maps = []
        logger.info("calclulating healpix maps")
        for m in match_data:
            match_data_hp_maps.append(pd.Series(hp.ang2pix(self.nside, m["ra"], m["dec"], lonlat=True), index=m.index))

        # Perform matching
        logger.info("matching ...")
        for primary_source_id in tqdm(primary_data.index.unique(), desc="primary sources"):
            i_primary_data = primary_data.loc[primary_source_id]
            primary_mean_ra = i_primary_data["ra"].median()
            primary_mean_dec = i_primary_data["dec"].median()
            primary_hp_indices = hp.ang2pix(self.nside, primary_mean_ra, primary_mean_dec, lonlat=True)

            # get secondary datapoints within disc
            datapoint_query_result = []
            for md, mh in zip(match_data, match_data_hp_maps):
                datapoint_query_result.append(md[mh == primary_hp_indices])
