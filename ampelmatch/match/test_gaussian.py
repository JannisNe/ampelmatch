import logging
from pathlib import Path

from ampelmatch.data.config import DatasetConfig
from ampelmatch.data.dataset import DatasetGenerator
from ampelmatch.data.plotter import Plotter
from ampelmatch.match.gaussian import StreamMatch


logger = logging.getLogger("ampelmatch.match.test_gaussian")


if __name__ == '__main__':
    logging.getLogger("ampelmatch").setLevel("DEBUG")
    dset_config_fname = Path(__file__).parent / "test_sim.json"
    dset_config = DatasetConfig.model_validate_json(dset_config_fname.read_text())
    dsets = DatasetGenerator(dset_config)
    if any([not f.exists() for f in dsets.filenames]):
        DatasetGenerator(dset_config).write()
        Plotter(dset_config).make_plots(n_lightcurves=10)

    batched_fns = []
    for i in range(1, dsets.n_transients + 1):
        batched_fns.append(dsets.filenames[:i*dsets.n_surveys])

    for i, fns in enumerate(batched_fns):
        logger.info(f"Matching batch {i}")
        match_config = {
            "name": f"{dset_config.name}_{i}",
            "plot": True,
            "nside": 128,
            "primary_data": {
                "filepath_or_buffer": fns[0],
                "index_col": 0
            },
            "match_data": [
                {
                    "filepath_or_buffer": fn,
                    "index_col": 0
                }
                for fn in fns[1:]
            ]
        }
        match = StreamMatch(**match_config)
        match.match()
