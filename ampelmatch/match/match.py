import functools
import logging
import abc
import numpy as np
import pandas as pd
import healpy as hp
from tqdm import tqdm
from pathlib import Path
from pydantic import BaseModel, computed_field, ConfigDict, model_validator, Field, PositiveInt, TypeAdapter
from typing import Any, Dict, Annotated, Union, Literal
from astropy.coordinates import angular_separation
import matplotlib.pyplot as plt
from matplotlib import cm, colors


logger = logging.getLogger(__name__)
SQARCSEC_TO_SR = np.radians(1 / 3600) ** 2


class BaseStreamMatch(BaseModel, abc.ABC):
    name: str
    match_type: str
    nside: int
    primary_data: dict
    match_data: list[dict]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    disc_radius_arcsec: float | None = 100
    plot: bool | PositiveInt = False
    plot_dir: Path | None = None
    markers: list[str] = ["o", "s", "x", "+", "d", "v", "^", "<", ">", "1", "2", "3", "4", "8", "p", "P", "*", "h", "H", "X"]
    cmaps: list[str] = ["viridis", "plasma", "inferno", "magma", "cividis", "twilight", "twilight_shifted", "turbo", "nipy_spectral"]

    @model_validator(mode='before')
    def plots_update(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get('plot_dir') is None:
            values['plot_dir'] = Path(values["name"]) / "plots"
        return values

    @computed_field
    def _primary_data(self) -> pd.DataFrame:
        logger.info(f"Loading primary data")
        logger.debug(f"primary data config: {self.primary_data}")
        return pd.read_csv(**self.primary_data)

    @computed_field
    def _match_data(self) -> list[pd.DataFrame]:
        logger.info("Loading match data")
        logger.debug(f"match data config: {self.match_data}")
        return [pd.read_csv(**d) for d in self.match_data]

    def get_pixels_disc(self, ra, dec):
        vec = hp.ang2vec(ra, dec, lonlat=True)
        return hp.query_disc(nside=self.nside, vec=vec, radius=np.radians(self.disc_radius_arcsec / 3600))

    def get_pixel(self, ra, dec):
        return [hp.ang2pix(self.nside, ra, dec, lonlat=True)]

    @abc.abstractmethod
    def calculate_bayes_factors(
            self,
            primary_ra: float, primary_dec: float, primary_data: pd.DataFrame,
            orig_sources: pd.DataFrame
    ) -> pd.Series:
        ...

    def disc_selection(self, match_data, ra, dec):
        match_data_hp_maps = []
        logger.info("calculating healpix maps")
        for m in match_data:
            match_data_hp_maps.append(
                pd.Series(
                    m.index,
                    index=hp.ang2pix(self.nside, m["ra"], m["dec"], lonlat=True)
                )
            )

        if (r := hp.pixelfunc.nside2resol(self.nside, arcmin=True)) < self.disc_radius_arcsec / 60:
            logger.debug("healpix resolution is better than disc radius, using query_disc")
            primary_hp_index = self.get_pixels_disc(ra, dec)
        else:
            logger.debug(
                f"healpix resolution {r} arcmin is worse than "
                f"disc radius {self.disc_radius_arcsec} arcsec, using only primary pixel"
            )
            primary_hp_index = self.get_pixel(ra, dec)

        selected_data = []
        for m, mh in zip(match_data, match_data_hp_maps):
            try:
                hpi = mh.loc[primary_hp_index]
            except KeyError:
                continue
            logger.debug(f"selected {len(hpi)} sources")
            selected_data.append(m.loc[hpi])
        return selected_data

    def match(self):
        logger.info("Matching streams")
        primary_data = self._primary_data
        match_data = self._match_data
        n_secondary = len(match_data)

        # Perform matching
        logger.info("matching ...")
        primary_source_bayes_factors = {}
        for primary_source_id in tqdm(primary_data.index.unique(), desc="primary sources"):
            i_primary_data = primary_data.loc[primary_source_id]
            primary_mean_ra = i_primary_data["ra"].median()
            primary_mean_dec = i_primary_data["dec"].median()

            if self.plot:
                gs = {"width_ratios": [1] + n_secondary * [0.1]}
                fig, axs = plt.subplots(ncols=n_secondary + 1, gridspec_kw=gs)
                ax = axs[0]
                ax.scatter(i_primary_data["ra"], i_primary_data["dec"], c="r", label="primary")

            if self.disc_radius_arcsec is not None:
                selected_match_data = self.disc_selection(match_data, primary_mean_ra, primary_mean_dec)
            else:
                selected_match_data = match_data

            # get secondary datapoints within disc
            bayes_factors = {}
            for imd, md in enumerate(selected_match_data):
                bayes_factors[imd] = self.calculate_bayes_factors(primary_mean_ra, primary_mean_dec, i_primary_data, md)

                if self.plot:

                    orig_sources = md.loc[bayes_factors[imd].index]
                    orig_sources.loc[:, "marker"] = "s"
                    orig_sources.loc[:, "woe"] = np.log10(bayes_factors[imd])
                    norm = colors.Normalize(vmin=min(orig_sources.woe), vmax=max(orig_sources.woe))
                    sm = cm.ScalarMappable(norm=norm, cmap=self.cmaps[imd])

                    for ii, i in enumerate(orig_sources.index.unique()):
                        ax.scatter(
                            orig_sources.loc[i, "ra"], orig_sources.loc[i, "dec"],
                            c=sm.to_rgba(orig_sources.loc[i, "woe"]),
                            label=f"match {ii}", marker=self.markers[ii % len(self.markers)]
                        )
                    plt.colorbar(sm, cax=axs[imd + 1], label=f"WOE {imd}")

            primary_source_bayes_factors[primary_source_id] = bayes_factors

            if self.plot:
                ax.set_aspect("equal")
                ax.set_xlabel("ra")
                ax.set_ylabel("dec")
                ax.legend()
                ax.legend()
                fname = self.plot_dir / f"{primary_source_id}.pdf"
                fname.parent.mkdir(exist_ok=True, parents=True)
                fig.savefig(fname)
                logger.debug(f"saved plot to {fname}")
                plt.close()

                if isinstance(self.plot, int):
                    self.plot -= 1

        return primary_source_bayes_factors


class GaussianStreamMatch(BaseStreamMatch):
    match_type: Literal["gaussian"]

    def calculate_bayes_factors(
            self,
            primary_ra: float, primary_dec: float, primary_data: pd.DataFrame,
            orig_sources: pd.DataFrame
    ) -> pd.Series:
        psi_rad = angular_separation(*[
            np.radians(v) for v in
            [primary_mean_ra, primary_mean_dec, md["ra"], md["dec"]]
        ])
        psi_arcsec = np.degrees(psi_rad) * 3600

        if self.disc_radius_arcsec is not None:
            m = psi_arcsec < self.disc_radius_arcsec
            n_within_disc = m.sum()
            logger.debug(f"{n_within_disc} within disc")
            orig_sources = md[m]

        sigmas_arcsec = orig_sources["sigma_arcsec"]
        primary_sigma_arcsec = primary_data["sigma_arcsec"].median()
        ssum = primary_sigma_arcsec ** 2 + sigmas_arcsec ** 2
        return 2 / ssum * np.exp(-psi_arcsec ** 2 / (2*ssum)) / SQARCSEC_TO_SR


class IceCubeContourStreamMatch(BaseStreamMatch):
    match_type: Literal["icecube_contour"]

    @staticmethod
    @functools.cache
    def contour_pixels_indices(filename):
        s, h = hp.read_map(filename, h=True)
        h = dict(h)
        if "Wilks theorem" in h["COMMENTS"]:
            llh_level = 4.605170185988092
        else:
            llh_level = 64.2
        ctr_pix = np.where(s < llh_level)[0]
        ctr_area = len(ctr_pix) * hp.nside2pixarea(h["NSIDE"])
        logger.debug(f"{filename}: {ctr_area} sr")
        return ctr_pix, ctr_area

    def calculate_bayes_factors(
            self,
            primary_ra: float, primary_dec: float, primary_data: pd.DataFrame,
            orig_sources: pd.DataFrame
    ) -> pd.Series:
        primary_sigma_arcsec = primary_data["sigma_arcsec"].median()
        filenames = orig_sources["filename"]
        pix = hp.ang2pix(self.nside, primary_ra, primary_dec, lonlat=True)
        bayes_factors = pd.Series(0.0, index=orig_sources.index)
        for i, r in orig_sources.iterrows():
            indices, area = self.contour_pixels_indices(r["filename"])
            if pix in indices:
                bayes_factors.loc[i] = 0.9 * (4 * np.pi) / area
            else:
                bayes_factors.loc[i] = 0.1 / (1 - area / (4 * np.pi))
        return bayes_factors


StreamMatch = TypeAdapter(Union[GaussianStreamMatch, IceCubeContourStreamMatch])
