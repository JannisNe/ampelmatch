import logging
import skysurvey
from cachier import cachier
import itertools

from ampelmatch.cache import cache_dir
from ampelmatch.data.config import DatasetConfig
from ampelmatch.data.positional_dataset import PositionalDataset
from ampelmatch.data.transients import TransientGenerator
from ampelmatch.data.surveys import SurveyGenerator


logger = logging.getLogger(__name__)


class DatasetGenerator:

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.surveys = SurveyGenerator(config.surveys)
        self.transients = TransientGenerator(config.transients)
        self.iter_configs = itertools.product(self.transients, self.surveys)

    def __iter__(self):
        return self

    def __next__(self):
        transient, survey = next(self.iter_configs)
        data = self.realize_data(survey, transient)
        dset = PositionalDataset(survey=survey, targets=transient, data=data)
        return dset

    @staticmethod
    @cachier(cache_dir=cache_dir)
    def realize_data(survey: skysurvey.Survey, targets: skysurvey.Target):
        logger.info(f"Generating dataset")
        return PositionalDataset.from_targets_and_survey(targets, survey).data
