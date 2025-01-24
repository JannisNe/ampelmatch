import logging
import skysurvey
from cachier import cachier
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
        return dset

    @staticmethod
    @cachier(cache_dir=cache_dir, hash_func=model_hash)
    def realize_data(survey: skysurvey.Survey, targets: skysurvey.Target, transient_config: Transient, survey_config: Survey):
        logger.info(f"Generating dataset")
        return PositionalDataset.from_targets_and_survey(targets, survey).data
