import abc
import functools
import json
import logging
from pathlib import Path
from typing import Any, Dict, Union, Literal

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ampelmatch.cache import dataframe_hash
from astropy.coordinates import angular_separation, SkyCoord
from matplotlib import cm, colors
from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
    PositiveInt,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)
SQDG_TO_SR = np.radians(1) ** 2
SQARCSEC_TO_SR = np.radians(1 / 3600) ** 2


class BaseBayesFactor(BaseModel, abc.ABC):
    name: str
    match_type: str
    nside: int
    model_config = ConfigDict(arbitrary_types_allowed=True)

    disc_radius_arcsec: float | None = 100
    plot: bool | PositiveInt = False
    plot_indices: list[Any] | None = None
    plot_dir: Path | None = None
    markers: list[str] = [
        "o",
        "s",
        "x",
        "+",
        "d",
        "v",
        "^",
        "<",
        ">",
        "1",
        "2",
        "3",
        "4",
        "8",
        "p",
        "P",
        "*",
        "h",
        "H",
        "X",
    ]
    cmaps: list[str] = [
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "twilight",
        "twilight_shifted",
        "turbo",
        "nipy_spectral",
    ]

    @model_validator(mode="before")
    def plots_update(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("plot_dir") is None:
            values["plot_dir"] = Path(values["name"]) / "plots"
        return values

    def get_pixels_disc(self, ra, dec):
        vec = hp.ang2vec(ra, dec, lonlat=True)
        return hp.query_disc(
            nside=self.nside, vec=vec, radius=np.radians(self.disc_radius_arcsec / 3600)
        )

    def get_pixel(self, ra, dec):
        return list(hp.get_all_neighbours(self.nside, ra, dec, lonlat=True)) + [
            hp.ang2pix(self.nside, ra, dec, lonlat=True)
        ]

    @abc.abstractmethod
    def calculate_bayes_factors(
        self,
        primary_ra: float,
        primary_dec: float,
        primary_data: pd.DataFrame,
        orig_sources: pd.DataFrame,
    ) -> pd.Series: ...

    @abc.abstractmethod
    def setup_plot(
        self, primary_data: pd.DataFrame, n_secondary: int
    ) -> tuple[plt.Figure, plt.Axes, list[plt.Axes]]: ...

    @staticmethod
    def plot_data(
        ax: plt.Axes, orig_sources: pd.DataFrame, marker: str, c: str, label: str = ""
    ):
        ax.scatter(
            orig_sources.ra,
            orig_sources.dec,
            c=c,
            marker=marker,
            label=label,
            transform=ax.get_transform("world"),
        )

    def add_data_to_plot(
        self, ax, data, color_column, cmap, cbar_label, cax, cbar_lim=(-1, 1)
    ):
        norm = colors.Normalize(
            vmin=min(list(data[color_column]) + [cbar_lim[1]]),
            vmax=max(list(data[color_column]) + [cbar_lim[1]]),
        )
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)

        if "source_index" in data.columns:
            for si in data.source_index.unique():
                m = data.source_index == si
                self.plot_data(
                    ax,
                    data.loc[m],
                    self.markers[si % len(self.markers)],
                    sm.to_rgba(data.loc[m, color_column]),
                    f"source {si}",
                )
        else:
            self.plot_data(
                ax,
                data,
                self.markers[0],
                sm.to_rgba(data[color_column]),
            )

        plt.colorbar(sm, cax=cax, label=cbar_label)

    def finalize_plot(self, fig, ax, primary_source_id):
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

    def disc_selection(self, match_data, ra, dec):
        match_data_hp_maps = []
        logger.debug("calculating healpix maps for disc selection")
        for m in match_data:
            match_data_hp_maps.append(
                pd.Series(
                    m.index,
                    index=hp.ang2pix(self.nside, m["ra"], m["dec"], lonlat=True),
                )
            )

        if (
            r := hp.pixelfunc.nside2resol(self.nside, arcmin=True)
        ) < self.disc_radius_arcsec / 60:
            logger.debug(
                "healpix resolution is better than disc radius, using query_disc"
            )
            primary_hp_index = self.get_pmatchixels_disc(ra, dec)
        else:
            logger.debug(
                f"healpix resolution {r} arcmin is worse than "
                f"disc radius {self.disc_radius_arcsec} arcsec, using only primary pixel"
            )
            primary_hp_index = self.get_pixel(ra, dec)

        selected_data = []
        for m, mh in zip(match_data, match_data_hp_maps):
            pixels_in_search_region_with_sources = mh.index.intersection(
                primary_hp_index
            )
            if len(pixels_in_search_region_with_sources) == 0:
                continue
            hpi = mh.loc[pixels_in_search_region_with_sources]
            logger.debug(f"selected {len(hpi)} sources")
            selected_data.append(m.loc[hpi])

        return selected_data

    def evaluate(self, primary_data: pd.DataFrame, match_data: list[pd.DataFrame]):
        logger.info("Matching streams")
        n_secondary = len(match_data)

        # Perform matching
        logger.info("matching ...")
        primary_source_bayes_factors = {}

        if self.plot_indices is None:
            if self.plot:
                self.plot_indices = np.random.choice(
                    primary_data.index.unique(), self.plot, replace=False
                )
            else:
                self.plot_indices = []

        for primary_source_id in tqdm(
            primary_data.index.unique(), desc="primary sources"
        ):
            i_primary_data = primary_data.loc[primary_source_id]
            if isinstance(i_primary_data, pd.Series):
                primary_mean_ra = i_primary_data["ra"]
                primary_mean_dec = i_primary_data["dec"]
            else:
                primary_mean_ra = i_primary_data["ra"].median()
                primary_mean_dec = i_primary_data["dec"].median()

            if primary_source_id in self.plot_indices:
                fig, ax, axs = self.setup_plot(i_primary_data, n_secondary)

            if self.disc_radius_arcsec is not None:
                selected_match_data = self.disc_selection(
                    match_data, primary_mean_ra, primary_mean_dec
                )
            else:
                selected_match_data = match_data

            # get secondary datapoints within disc
            bayes_factors = {}
            for imd, md in enumerate(selected_match_data):
                bayes_factors[imd] = self.calculate_bayes_factors(
                    primary_mean_ra, primary_mean_dec, i_primary_data, md
                )
                if primary_source_id in self.plot_indices:
                    bf = bayes_factors[imd]
                    bf = bf[bf > 0]
                    orig_sources = md.loc[bf.index]
                    orig_sources.loc[:, "marker"] = "s"
                    orig_sources.loc[:, "woe"] = np.log10(bf)
                    self.add_data_to_plot(
                        ax, orig_sources, "woe", self.cmaps[imd], f"WOE {imd}", axs[imd]
                    )

            primary_source_bayes_factors[primary_source_id] = bayes_factors

            if primary_source_id in self.plot_indices:
                self.finalize_plot(fig, ax, primary_source_id)

        return primary_source_bayes_factors


class GaussianBayesFactor(BaseBayesFactor):
    match_type: Literal["gaussian"]

    def calculate_bayes_factors(
        self,
        primary_ra: float,
        primary_dec: float,
        primary_data: pd.DataFrame,
        orig_sources: pd.DataFrame,
    ) -> pd.Series:
        psi_rad = angular_separation(
            *[
                np.radians(v)
                for v in [
                    primary_ra,
                    primary_dec,
                    orig_sources["ra"],
                    orig_sources["dec"],
                ]
            ]
        )
        psi_arcsec = np.degrees(psi_rad) * 3600

        if self.disc_radius_arcsec is not None:
            m = psi_arcsec < self.disc_radius_arcsec
            n_within_disc = m.sum()
            logger.debug(f"{n_within_disc} within disc")
            orig_sources = orig_sources[m]
            psi_arcsec = psi_arcsec[m]

        sigmas_arcsec = orig_sources["sigma_arcsec"]
        primary_sigma_arcsec = primary_data["sigma_arcsec"].median()
        ssum = primary_sigma_arcsec**2 + sigmas_arcsec**2
        return 2 / ssum * np.exp(-(psi_arcsec**2) / (2 * ssum)) / SQARCSEC_TO_SR

    def setup_plot(
        self, primary_data: pd.DataFrame, n_secondary: int
    ) -> tuple[plt.Figure, plt.Axes, list[plt.Axes]]:
        fig = plt.figure()
        gridspec = fig.add_gridspec(
            ncols=n_secondary + 1, width_ratios=[1] + n_secondary * [0.1]
        )
        center = SkyCoord(
            primary_data["ra"].median(), primary_data["dec"].median(), unit="deg"
        )
        radius = self.disc_radius_arcsec * u.arcsec
        ax = fig.add_subplot(
            gridspec[:, 0],
            projection="astro degrees zoom",
            center=center,
            radius=radius,
        )
        axs = [fig.add_subplot(gridspec[:, i]) for i in range(1, n_secondary + 1)]
        ax.scatter(
            primary_data["ra"],
            primary_data["dec"],
            c="r",
            label="primary",
            transform=ax.get_transform("world"),
        )
        return fig, ax, axs


class IceCubeContourBayesFactor(BaseBayesFactor):
    match_type: Literal["icecube_contour"]
    disc_radius_arcsec: None = None
    contour_cache: dict = {}

    @staticmethod
    @functools.cache
    def contour_pixels_indices(filename: str | Path):
        filename = Path(filename).resolve()
        cache_file = filename.with_suffix(".cache")
        if not cache_file.exists():
            logger.debug(f"{cache_file} does not exist, calculating contour pixels")
            s, h = hp.read_map(filename, h=True)
            h = dict(h)
            if "Wilks theorem" in h["COMMENTS"]:
                llh_level = 4.605170185988092
            else:
                llh_level = 64.2
            ctr_pix = np.where(s < llh_level)[0]
            ctr_area = len(ctr_pix) * hp.nside2pixarea(h["NSIDE"])
            res_str = json.dumps(
                {
                    "ctr_pix": ctr_pix.tolist(),
                    "ctr_area": ctr_area,
                    "llh_level": llh_level,
                }
            )
            cache_file.write_text(res_str)
            logger.debug(f"saved contour pixels to {cache_file}")
        else:
            logger.debug(f"reading contour pixels from {cache_file}")
            res = json.loads(cache_file.read_text())
            ctr_pix = np.array(res["ctr_pix"])
            ctr_area = res["ctr_area"]
            llh_level = res["llh_level"]
        logger.debug(f"{filename}: {ctr_area / SQDG_TO_SR} sqd")
        return ctr_pix, ctr_area, llh_level

    def get_contour_cache(self, data):
        h = dataframe_hash(data)
        if h not in self.contour_cache:
            logger.info("making contour cache")
            nsides = data["nside"].unique()
            logger.debug(f"unique nsides: {nsides}")
            cache_dict = {}
            for nside in nsides:
                m = data["nside"] == nside
                logger.debug(f"nside {nside}: {m.sum()} sources")
                pixels = {i: [] for i in range(hp.nside2npix(nside))}
                values = pd.DataFrame(
                    index=data[m].index,
                    columns=["bayes_factor_in", "bayes_factor_out"],
                    dtype=float,
                )
                for i, r in data[m].iterrows():
                    ctr_pix, ctr_area, llh_level = self.contour_pixels_indices(
                        r["filename"]
                    )
                    values.loc[i, "bayes_factor_in"] = 0.9 * (4 * np.pi) / ctr_area
                    values.loc[i, "bayes_factor_out"] = 0.1 / (
                        1 - ctr_area / (4 * np.pi)
                    )
                    for p in ctr_pix:
                        pixels[p].append(i)
                cache_dict[nside] = (pixels, values)
            self.contour_cache[h] = cache_dict
        return self.contour_cache[h]

    def calculate_bayes_factors(
        self,
        primary_ra: float,
        primary_dec: float,
        primary_data: pd.DataFrame,
        orig_sources: pd.DataFrame,
    ) -> pd.Series:
        bayes_factors = pd.Series(0.0, index=orig_sources.index)
        contour_cache = self.get_contour_cache(orig_sources)
        pix_indices = np.atleast_1d(
            hp.ang2pix(list(contour_cache.keys()), primary_ra, primary_dec, lonlat=True)
        )
        for pix_index, (pixels, values) in zip(pix_indices, contour_cache.values()):
            in_indices = pixels[pix_index]
            out_indices = values.index.difference(in_indices)
            for i, s in zip([in_indices, out_indices], ["in", "out"]):
                if len(i) > 0:
                    logger.debug(f"{s}side {len(in_indices)} contours")
                    bayes_factors.loc[i] = values.loc[i, f"bayes_factor_{s}"]
        return bayes_factors

    def setup_plot(
        self, primary_data: pd.DataFrame, n_secondary: int
    ) -> tuple[plt.Figure, plt.Axes, list[plt.Axes]]:
        fig = plt.figure()
        gridspec = fig.add_gridspec(
            ncols=n_secondary + 1, width_ratios=[1] + n_secondary * [0.1]
        )
        if isinstance(primary_data, pd.Series):
            ra = primary_data["ra"]
            ra_med = ra
            dec = primary_data["dec"]
            dec_med = dec
        else:
            ra = primary_data["ra"].values
            ra_med = primary_data["ra"].mean()
            dec = primary_data["dec"].values
            dec_med = primary_data["dec"].mean()
        center = SkyCoord(ra_med, dec_med, unit="deg")
        ax = fig.add_subplot(
            gridspec[:, 0], projection="astro degrees mollweide", center=center
        )
        t = ax.get_transform("world")
        axs = [fig.add_subplot(gridspec[:, i]) for i in range(1, n_secondary + 1)]
        ax.scatter(ra, dec, c="r", label="primary", transform=t)
        return fig, ax, axs

    def plot_data(
        self,
        ax: plt.Axes,
        orig_sources: pd.DataFrame,
        marker: str,
        c: str,
        label: str = "",
    ):
        c = np.atleast_1d(c)
        for i, r in orig_sources.iterrows():
            fn = r["filename"]
            logger.debug(f"plotting contour {fn}")
            _, _, llh_level = self.contour_pixels_indices(fn)
            ax.contour_hpx(fn, levels=[llh_level], colors=[c[i]])


BayesFactor = Union[GaussianBayesFactor, IceCubeContourBayesFactor]
