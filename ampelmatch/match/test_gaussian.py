import importlib
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ampelmatch.data.config import DatasetConfig
from ampelmatch.data.dataset import DatasetGenerator
from ampelmatch.data.plotter import Plotter
from ampelmatch.data.surveys import SurveyGenerator
from ampelmatch.data.transients import TransientGenerator
from ampelmatch.match import match
from networkx.algorithms.bipartite import density

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

    a = min(s.fov * len(s.fields) for s in dset_config.surveys)
    for i, fns in enumerate(batched_fns):
        logger.info(f"Matching batch {i}")
        match_config = {
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
            "bayes_factor": {
                "name": f"{dset_config.name}/{h}/{i}_fine",
                "match_type": "gaussian",
                "plot": 10,
                "nside": 1024,
            },
            "prior": {
                "name": "surface_density",
                "nside": 128,
                "area_sqdg": a,
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
            "posterior_threshold": 0.95,
        }
        matcher = match.StreamMatch.model_validate(match_config)
        probabilities = matcher.posteriors
        primary_data = pd.read_csv(**matcher.primary_data)
        match_data = [pd.read_csv(**d) for d in matcher.match_data]
        matches = matcher.match()
        eff = []
        pur = []
        for j in range(len(match_data)):
            jmd = match_data[j]
            jeff = []
            jpur = []
            for sid in primary_data.index.unique():
                true_ids = jmd[jmd["source_index"] == sid].index
                matched_ids = matches[sid][j]
                good_matches = true_ids.intersection(matched_ids)
                bad_matches = set(matched_ids) - set(true_ids)
                jeff.append(len(good_matches) / len(true_ids))
                jpur.append(len(bad_matches) / len(matched_ids))
            eff.append(jeff)
            pur.append(jpur)

        print(
            matcher.n_matches(),
            matcher.posterior_sum(),
            len(match_data[0]),
            [np.quantile(e, [0.05, 0.5, 0.95]) for e in eff],
            [np.quantile(p, [0.05, 0.5, 0.95]) for p in pur],
        )

        fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharey=True)
        for e, p in zip(eff, pur):
            axs[0].hist(e, density=True)
            axs[1].hist(p, density=True)
        axs[0].set_ylabel("density")
        axs[0].set_xlabel("efficiency")
        axs[0].set_xlabel("purity")
        fig.savefig("performance.pdf")
        plt.close()
