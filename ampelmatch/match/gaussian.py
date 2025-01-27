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
    disc_radius_arcsec: float = 100
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
        logger.info("calculating healpix maps")
        for m in match_data:
            match_data_hp_maps.append(
                pd.Series(
                    m.index,
                    index=hp.ang2pix(self.nside, m["ra"], m["dec"], lonlat=True)
                )
            )

        if hp.pixelfunc.nside2resol(self.nside, arcmin=True) < self.disc_radius_arcsec / 60:
            logger.debug("healpix resolution is smaller than disc radius, using query_disc")
            _get_pixels = self.get_pixels_disc
        else:
            logger.debug("healpix resolution is larger than disc radius, using only primary pixel")
            _get_pixels = self.get_pixel

        # Perform matching
        logger.info("matching ...")
        for primary_source_id in tqdm(primary_data.index.unique(), desc="primary sources"):
            i_primary_data = primary_data.loc[primary_source_id]
            primary_mean_ra = i_primary_data["ra"].median()
            primary_mean_dec = i_primary_data["dec"].median()
            primary_hp_index = _get_pixels(primary_mean_ra, primary_mean_dec)
            # logger.debug(f"primary hp index: {primary_hp_index}")

            # get secondary datapoints within disc
            datapoint_query_result = []
            for md, mh in zip(match_data, match_data_hp_maps):
                try:
                    dps = mh.loc[primary_hp_index]
                except KeyError:
                    dps = []
                # logger.debug(f"{len(dps)} within first search")
                datapoint_query_result.append(dps)

    def get_pixels_disc(self, ra, dec):
        vec = hp.ang2vec(ra, dec, lonlat=True)
        return hp.query_disc(nside=32, vec=vec, radius=np.radians(self.disc_radius_arcsec / 3600))

    def get_pixel(self, ra, dec):
        return [hp.ang2pix(self.nside, ra, dec, lonlat=True)]
