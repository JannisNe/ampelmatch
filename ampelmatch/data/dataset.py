import logging
import skysurvey
import diskcache
import itertools

from ampelmatch import cache_dir
from ampelmatch.data.config import DatasetConfig
from ampelmatch.data.positional_dataset import PositionalDataset
from ampelmatch.data.transients import TransientGenerator
from ampelmatch.data.surveys import SurveyGenerator


logger = logging.getLogger(__name__)
cache = diskcache.Cache(cache_dir)


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
        logger.info(f"Generating dataset for {survey.name} and {transient.name}")
        data = self.realize_data(survey, transient)
        dset = PositionalDataset(survey, transient, data)
        return dset

    @staticmethod
    @cache.memoize()
    def realize_data(survey: skysurvey.Survey, targets: skysurvey.Target):
        return PositionalDataset.from_targets_and_survey(targets, survey).data
