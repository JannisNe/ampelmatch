import logging
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from pathlib import Path
from astropy.time import Time

from ampelmatch.data.config import DatasetConfig
from ampelmatch.data.dataset import DatasetGenerator


logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, config: DatasetConfig):
        self.datasets = DatasetGenerator(config)
        self.config = config
        self.dir = Path(config.name)
        self.dir.mkdir(exist_ok=True)
        self.batch_size = self.datasets.n_surveys

    def batched(self):
        dsets = []
        for d, c in self.datasets:
            dsets.append(d)
            if len(dsets) == self.batch_size:
                yield dsets
                dsets = []

    def make_data_plots(self, skyplot=True, n_lightcurves=10):
        for i, dsets in enumerate(self.batched()):
            logger.info(f"Generating plots for dataset {i}")
            n_det = [d.get_ndetection() for d in dsets]

            if skyplot:
                surveys = [d.surveys for d in dsets]
                targets = [d.targets for d in dsets]
                fig, ax = self.sky_plot(surveys, targets, n_det)
                fn = self.dir / f"sky_coverage_{i}.pdf"
                fig.savefig(fn)
                logger.info(f"Saved sky coverage plot to {fn}")
                plt.close()

            indices = list(set.intersection(*[set(n.index) for n in n_det]))
            for j in np.random.choice(indices, n_lightcurves):
                logger.info(f"Plotting target {j}")
                fig, axs = self.lightcurve_plot(dsets, j)
                fn = self.dir / f"lightcurve_{i}_{j}.pdf"
                fig.savefig(fn)
                logger.info(f"Saved lightcurve plot to {fn}")
                plt.close()

    @staticmethod
    def lightcurve_plot(dsets, i):
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        for il, l in enumerate(dsets):
            lc = l.get_target_lightcurve(index=i)
            zp = 25
            coef = 10 ** (-(lc["zp"] - zp) / 2.5)
            lc["flux_zp"] = lc["flux"] * coef
            lc["fluxerr_zp"] = lc["fluxerr"] * coef
            bands = np.unique(lc["band"])
            c = [f"C{il}"] * len(bands)
            l.targets.show_lightcurve(bands, ax=ax1, fig=fig, index=i,
                                      format_time=True, t0_format="mjd",
                                      zp=zp, colors=c,
                                      zorder=2)
            for iband, (band_, color_) in enumerate(zip(bands, c)):
                if color_ is None:
                    ecolor = to_rgba("0.4", 0.2)
                else:
                    ecolor = to_rgba(color_, 0.2)

                obs_band = lc[lc["band"] == band_]
                times = Time(obs_band["time"], format="mjd").datetime
                ax1.scatter(times, obs_band["flux_zp"], color=color_, zorder=4,
                            label=f"Survey {il}" if iband == 0 else None)
                ax1.errorbar(times, obs_band["flux_zp"],
                             yerr=obs_band["fluxerr_zp"],
                             ls="None", marker="None", ecolor=ecolor,
                             zorder=3)

            ax2.scatter(lc["ra"], lc["dec"], label=f"Survey {il}")
        ax1.legend()
        ax2.legend()
        ax2.set_aspect("equal")

        return fig, (ax1, ax2)

    @staticmethod
    def sky_plot(surveys, targets, n_det):
        logger.info(f"Plotting sky coverage")
        origin = 180
        t = ccrs.PlateCarree(central_longitude=origin)
        fig = plt.figure()
        ax = fig.add_axes((0.15, 0.22, 0.75, 0.75), projection=ccrs.Mollweide())
        for dddi, (s, targets) in enumerate(zip(surveys, targets)):
            geodf = s.fields.copy()
            xy = np.stack(geodf["geometry"].apply(lambda x: ((np.asarray(x.exterior.xy)).T)).values)
            # correct edge effects
            flag_egde = np.any(np.diff(xy, axis=1) > 300, axis=1)[:, 0]
            xy[flag_egde] = ((xy[flag_egde] + origin) % 360 - origin)
            geodf["xy"] = list(xy)
            ax.add_collection(PolyCollection(
                geodf["xy"], transform=t, ec=f"C{dddi}", label=f"Survey {dddi}", alpha=0.5, fc="none"
            ))
            det_ids = n_det[dddi].index
            if targets.data is not None:
                det = targets.data.loc[det_ids]
                ax.scatter(det["ra"], det["dec"], transform=t, color=f"C{dddi}", s=1)

        ax.autoscale()
        ax.set_global()
        ax.legend()
        ax.gridlines()
        return fig, ax
