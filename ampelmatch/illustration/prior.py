import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("ampelmatch.illustration.prior")
SQARCSEC_TO_SR = np.radians(1 / 3600) ** 2


def prior_plot():
    sigma_1 = 0.1  # arcsec
    sigma_2 = 0.2  # arcsec
    ssum = sigma_1**2 + sigma_2**2
    psi = np.linspace(0, 3, 100)  # arcsec
    b = 2 / ssum * np.exp(-(psi**2) / (2 * ssum)) / SQARCSEC_TO_SR
    posts = []
    simple_posts = []
    densities = [1e4, 1e5, 1e6, 1e11]
    for density in densities:
        prior = 1 / (density * (180 / np.pi) ** 2 * 4 * np.pi)
        logger.info(f"Prior: {prior}")
        posts.append((1 + (1 - prior) / (prior * b)) ** -1)
        simple_posts.append(b * prior / (1 + b * prior))

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(psi, np.log10(b), label=r"$\log _{10}(B)$", color="blue")
    for p, sp, d, ls in zip(posts, simple_posts, densities, ["-", "--", ":", "-."]):
        ax2.plot(psi, sp, label=f"{d:.0e}", color="red", linestyle=ls)
    ax.set_xlabel("Separation (arcsec)")
    ax.set_ylabel(r"$\log _{10}(B)$")
    ax2.set_ylabel("Posterior")
    ax2.set_ylim(0, 1)
    ax2.axhline(0.9, color="red", linestyle="--")
    ax.set_ylim(-2, 13)
    ax.legend(loc="lower left")
    ax2.legend(loc="lower right", title="Prior [deg$^{-2}$]")
    fn = Path("prior.pdf").resolve()
    logger.info(f"Saving plot to {fn}")
    fig.savefig(fn)
    plt.close()


if __name__ == "__main__":
    logging.getLogger("ampelmatch").setLevel(logging.INFO)
    prior_plot()
