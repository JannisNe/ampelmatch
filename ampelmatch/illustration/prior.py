import logging
from pathlib import Path

import ampelmatch
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("ampelmatch.illustration.prior")
SQARCSEC_TO_SR = np.radians(1 / 3600) ** 2


def prior_plot():

    area = 2.0  # sq deg
    sigma_1 = 1  # arcsec
    sigma_2 = 6  # arcsec
    ssum = sigma_1**2 + sigma_2**2
    psi = np.linspace(0, 100, 100)  # arcsec
    b = 2 / ssum * np.exp(-(psi**2) / (2 * ssum)) / SQARCSEC_TO_SR
    posts = []
    densities = [1e5, 1e6, 1e7, 8.5e11]
    for density in densities:
        prior = 1 / (density * (np.pi / 180) ** 2 * 4 * np.pi)
        logger.info(f"Prior: {prior}")
        posts.append((1 + (1 - prior) / (prior * b)) ** -1)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(psi, np.log10(b), label="WOE", color="blue")
    for p, d, ls in zip(posts, densities, ["-", "--", ":", "-."]):
        ax2.plot(psi, p, label=f"{d:.0e} / sqdg", color="red", linestyle=ls)
    ax.set_xlabel("Separation (arcsec)")
    ax.set_ylabel(r"WOE")
    ax2.set_ylabel("Posterior")
    ax2.set_ylim(0, 1)
    ax.set_ylim(-2, 13)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fn = Path("prior.pdf").resolve()
    logger.info(f"Saving plot to {fn}")
    fig.savefig(fn)
    plt.close()


if __name__ == "__main__":
    logging.getLogger("ampelmatch").setLevel(logging.INFO)
    prior_plot()
