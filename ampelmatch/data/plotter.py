import logging
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from pathlib import Path

from ampelmatch.data.config import DatasetConfig
from ampelmatch.data.dataset import DatasetGenerator


logger = logging.getLogger(__name__)


class Plotter:
    def __init__(self, config: DatasetConfig):
        self.datasets = DatasetGenerator(config)
        self.config = config
        self.dir = Path(config.name)
        self.batch_size = len(self.config.surveys)

    def batched(self):
        dsets = []
        for d in self.datasets:
            dsets.append(d)
            if len(dsets) == self.batch_size:
                yield dsets
                dsets = []

    def make_plots(self):
        n_surveys = len(self.config.surveys)
        for i, dsets in enumerate(self.batched()):
            logger.info(f"Generating plots for dataset {i}")
            n_det = [d.get_ndetection() for d in dsets]
            fig, ax = self.sky_plot(dsets, n_det)
            fn = self.dir / f"sky_coverage_{i}.pdf"
            fig.savefig(fn)
            logger.info(f"Saved sky coverage plot to {fn}")
            plt.close()

            indices = list(set.intersection(*[set(n.index) for n in n_det]))
            for j in np.random.choice(indices, 10):
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
            l.show_target_lightcurve(index=i, ax=ax1, label=f"Survey {il}")
        ax1.legend()
        for il, l in enumerate(dsets):
            lc = l.get_target_lightcurve(index=i)
            ax2.scatter(lc["ra"], lc["dec"], label=f"Survey {il}")
        ax2.legend()
        ax2.set_aspect("equal")

        return fig, (ax1, ax2)

    @staticmethod
    def sky_plot(dsets, n_det):
        logger.info(f"Plotting sky coverage for transient {j}")
        origin = 180
        t = ccrs.PlateCarree(central_longitude=origin)
        fig = plt.figure()
        ax = fig.add_axes((0.15, 0.22, 0.75, 0.75), projection=ccrs.Mollweide())
        for dddi, ddd in enumerate([dsets]):
            s = ddd.survey
            data = s.get_fieldstat(stat="size", columns=None, incl_zeros=True, fillna=np.nan, data=None)
            geodf = s.fields.copy()
            xy = np.stack(geodf["geometry"].apply(lambda x: ((np.asarray(x.exterior.xy)).T)).values)
            # correct edge effects
            flag_egde = np.any(np.diff(xy, axis=1) > 300, axis=1)[:, 0]
            xy[flag_egde] = ((xy[flag_egde] + origin) % 360 - origin)
            geodf["xy"] = list(xy)
            ax.add_collection(PolyCollection(
                geodf["xy"], transform=t, ec=f"C{dddi}", label=f"Survey {dddi}", alpha=0.5, fc="none"
            ))
            targets = ddd.targets
            det_ids = n_det[dddi].index
            det = targets.data.loc[det_ids]
            ax.scatter(det["ra"], det["dec"], transform=t, color=f"C{dddi}", s=1)

        # ax.autoscale()
        ax.set_global()
        ax.legend()
        ax.gridlines()
        return fig, ax
