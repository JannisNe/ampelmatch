import logging
from pathlib import Path

from ampelmatch.data.config import DatasetConfig
from ampelmatch.data.dataset import DatasetGenerator
from ampelmatch.match.gaussian import StreamMatch


if __name__ == '__main__':
    logging.getLogger("ampelmatch").setLevel("DEBUG")
    dset_config_fname = Path(__file__).parent / "test_sim.json"
    dset_config = DatasetConfig.model_validate_json(dset_config_fname.read_text())
    DatasetGenerator(dset_config).write()
