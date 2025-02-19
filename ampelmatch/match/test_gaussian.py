import importlib
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ampelmatch.data.config import DatasetConfig
from ampelmatch.data.dataset import DatasetGenerator
from ampelmatch.data.plotter import Plotter
from ampelmatch.data.surveys import SurveyGenerator
from ampelmatch.data.transients import TransientGenerator
from ampelmatch.match import match

match = importlib.reload(match)

logger = logging.getLogger("ampelmatch.match.test_gaussian")


if __name__ == "__main__":
    logging.getLogger("ampelmatch").setLevel("INFO")
    dset_config_fname = Path(__file__).parent / "test_sim.json"
    dset_config = DatasetConfig.model_validate_json(dset_config_fname.read_text())
    h = dset_config.get_hash()
    for t, tc in zip(
        TransientGenerator(dset_config.transients), dset_config.transients
    ):
        surveys = list(SurveyGenerator(dset_config.surveys))
        targets = [t for _ in surveys]
        fig, ax = Plotter.sky_plot(surveys, targets, target_skyarea=tc.skyarea)
        fname = Path(dset_config.name) / h / f"{tc.transient_type}_sky_coverage.pdf"
        logger.info(f"Saving sky coverage plot to {fname}")
        fname.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(fname)
        plt.close()

    dsets = DatasetGenerator(dset_config)
    if any([not f.exists() for f in dsets.filenames]):
        DatasetGenerator(dset_config).write()
        Plotter(dset_config).make_data_plots(n_lightcurves=10)

    batched_fns = []
    for i in range(1, dsets.n_transients + 1):
        batched_fns.append(dsets.filenames[: i * dsets.n_surveys])

    for i, fns in enumerate(batched_fns):
        logger.info(f"Matching batch {i}")
        match_config = {
            "bayes_factor": {
                "name": f"{dset_config.name}/{h}/{i}_fine",
                "match_type": "gaussian",
                "plot": 10,
                "nside": 1024,
                "primary_data": {
                    "filepath_or_buffer": fns[0],
                    "index_col": 0,
                    # "nrows": 700,
                },
                "match_data": [
                    {
                        "filepath_or_buffer": fn,
                    }
                    for fn in fns[1:]
                ],
            },
            "prior": {
                "name": "surface_density",
                "nside": 128,
                "primary_data": {
                    "filepath_or_buffer": fns[0],
                    "index_col": 0,
                    # "nrows": 700,
                },
                "match_data": [
                    {
                        "filepath_or_buffer": fn,
                    }
                    for fn in fns[1:]
                ],
            },
        }
        match = match.StreamMatch.model_validate(match_config)
        prod = sum([len(d) for d in match.bayes_factor.match_data_df])
        prios_values = [
            match.prior(match.bayes_factor.primary_data_df.loc[dd])
            for dd in match.bayes_factor.primary_data_df.index.unique()
        ]
        a = min(s.fov for s in dset_config.surveys) * (np.pi / 180) ** 2
        print(
            sum(prios_values)
            # * prod
            * (a / (4 * np.pi)) ** (1 - len(match.bayes_factor.match_data_df)),
            prod,
        )
        # probabilities = match.calculate_posteriors()
