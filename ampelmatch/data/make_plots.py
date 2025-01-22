import logging

import matplotlib.pyplot as plt
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
                try:
                    l[j].show_target_lightcurve(index=i, ax=ax, label=f"Survey {il}")
                except KeyError:
                    logger.info(f"Survey {j} does not have target {i}")
            ax.legend()
            fig.savefig(f"target_{j}_{i}.pdf")
            plt.close()
            n_plotted += 1
            i += 1
