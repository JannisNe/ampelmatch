import logging
import skysurvey
from cachier import cachier
from pathlib import Path
import itertools

from ampelmatch.cache import cache_dir, model_hash
from ampelmatch.data.config import DatasetConfig, Survey, Transient
from ampelmatch.data.positional_dataset import PositionalDataset
from ampelmatch.data.transients import TransientGenerator
from ampelmatch.data.surveys import SurveyGenerator


logger = logging.getLogger(__name__)


class DatasetGenerator:

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.surveys = SurveyGenerator(config.surveys)
        self.transients = TransientGenerator(config.transients)
        self.iter = itertools.product(self.transients, self.surveys)
        self.iter_configs = itertools.product(config.transients, config.surveys)

    def __iter__(self):
        return self

    def __next__(self):
        transient, survey = next(self.iter)
        configs = next(self.iter_configs)
        data = self.realize_data(survey, transient, *configs)
        dset = PositionalDataset(survey=survey, targets=transient, data=data)
        return dset, configs

    @staticmethod
    @cachier(cache_dir=cache_dir, hash_func=model_hash)
    def realize_data(survey: skysurvey.Survey, targets: skysurvey.Target, transient_config: Transient, survey_config: Survey):
        logger.info(f"Generating dataset")
        return PositionalDataset.from_targets_and_survey(targets, survey).data

    @property
    def filenames(self) -> list[Path]:
        directory = Path(self.config.name)
        return [
            directory / self.config.get_hash() / f"{s.name}_{t.transient_type}.csv"
            for t, s in itertools.product(self.config.transients, self.config.surveys)
        ]

    @property
    def n_transients(self):
        return len(self.config.transients)

    @property
    def n_surveys(self):
        return len(self.config.surveys)

    def write(self):
        for (d, confs), fname in zip(self, self.filenames):
            fname.parent.mkdir(exist_ok=True, parents=True)
            d.data.index.names = ["source_index", "detection_index"]
            d.data.to_csv(fname)
            logger.info(f"saved {fname}")
