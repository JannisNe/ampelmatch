import logging
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from ampelmatch.data.dataset import generate_test_data


logger = logging.getLogger("ampelmatch.data.make_plots")


if __name__ == '__main__':
    logging.getLogger("ampelmatch").setLevel("INFO")
    logging.getLogger("ampelmatch").info("Generating test data")
    d = list(generate_test_data())
    n_det = [[ddd.get_ndetection() for ddd in dd] for dd in d]
    max_i = [max(indet.index) for ndet in n_det for indet in ndet]

    i = 0
    n_plotted = 0
    for j in range(len(d[0])):
        while n_plotted < 3:
            while not all([i in _ndet[j] for _ndet in n_det]):
                i += 1
            logger.info(f"Plotting target {i}")

            fig, ax = plt.subplots()
            for il, l in enumerate(d):
                l[j].show_target_lightcurve(index=i, ax=ax, label=f"Survey {il}")
            ax.legend()
            fig.savefig(f"target_{j}_{i}.pdf")
            plt.close()

            fig, ax = plt.subplots()
            for il, l in enumerate(d):
                lc = l[j].get_target_lightcurve(index=i)
                ax.scatter(lc["ra"], lc["dec"], label=f"Survey {il}")
            ax.legend()
            ax.set_aspect("equal")
            fig.savefig(f"target_{j}_{i}_pos.pdf")
            plt.close()

            n_plotted += 1
            i += 1

        logger.info(f"Plotting sky coverage for transient {j}")
        origin = 180
        t = ccrs.PlateCarree(central_longitude=origin)
        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.22, 0.75, 0.75], projection=ccrs.Mollweide())
        for dddi, ddd in enumerate([dd[j] for dd in d]):
            s = ddd.survey
            data = s.get_fieldstat(stat="size", columns=None, incl_zeros=True, fillna=np.nan, data=None)
            geodf = s.fields.copy()
            xy = np.stack(geodf["geometry"].apply(lambda x: ((np.asarray(x.exterior.xy)).T)).values)
            # correct edge effects
            flag_egde = np.any(np.diff(xy, axis=1) > 300, axis=1)[:, 0]
            xy[flag_egde] = ((xy[flag_egde] + origin) % 360 - origin)
            geodf["xy"] = list(xy)
            ax.add_collection(PolyCollection(
                geodf["xy"], transform=t, color=f"C{dddi}", label=f"Survey {dddi}", alpha=0.5, edgecolor="none"
            ))
            targets = ddd.targets
            det_ids = n_det[dddi][j].index
            det = targets.data.loc[det_ids]
            ax.scatter(det["ra"], det["dec"], transform=t, color=f"C{dddi}")

        # ax.autoscale()
        ax.set_global()
        ax.legend()
        ax.gridlines()
        fig.savefig(f"sky_{j}.pdf")
        plt.close()
