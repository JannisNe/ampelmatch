import logging

from ampelmatch.data.config import Survey
from ampelmatch.data.positional_survey import PositionalGridSurvey


logger = logging.getLogger(__name__)


class SurveyGenerator:

    survey_classes = [PositionalGridSurvey]
    survey_dict = {cls.__name__: cls for cls in survey_classes}

    def __init__(self, configs: list[Survey]):
        self.configs = configs
        self.iter_config = iter(configs)

    def __iter__(self):
        logger.info("generating test surveys")
        return self

    def __next__(self):
        config = next(self.iter_config)
        logger.debug(f"generating survey {config}")
        return self.survey_dict[config.type].from_config(config)
