import ampelmatch
import matplotlib.pyplot as plt
import logging


logger = logging.getLogger("ampelmatch.illustration.nearest_neighbour")


catalog_style = {
    "Survey1": {"color": "orange", "marker": "o"},
    "Survey2": {"color": "blue", "marker": "s"},
}


def make_plot(
        coords: list[tuple[float, float]],
        sigmas: list[float],
        catalogs: list[str],
        fname: str
):
    fig, ax = plt.subplots()
    for coord, sigma, catalog in zip(coords, sigmas, catalogs):
        ax.scatter(*coord, **catalog_style[catalog])
        ax.add_artist(plt.Circle(coord, sigma, fill=False, color=catalog_style[catalog]["color"]))

    for cat, style in catalog_style.items():
        ax.scatter([], [], label=cat, **style)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlabel(r"$\Delta$RA")
    ax.set_ylabel(r"$\Delta$Dec")
    ax.set_xlim(-.5, .5)
    ax.set_ylim(-.5, .5)

    logger.info(f"Saving plot to {fname}")
    fig.savefig(fname)
    plt.close()


if __name__ == '__main__':
    logging.getLogger("ampelmatch").setLevel(logging.DEBUG)
    make_plot(
        [(0, 0), (.2, .1), (.4, .4), (-.4, -.3)],
        [0.1, 0.2, 0.1, .22],
        ["Survey1", "Survey2", "Survey1", "Survey2"],
        "nearest_neighbour_match.pdf"
    )
    make_plot(
        [(0, 0), (.2, .1), (.18, .11), (-.4, -.3)],
        [0.1, 0.2, 0.1, .22],
        ["Survey1", "Survey2", "Survey1", "Survey2"],
        "nearest_neighbour_nomatch.pdf"
    )
