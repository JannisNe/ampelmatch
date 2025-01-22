import logging
import skysurvey

from ampelmatch.data.positional_uncertainty import BaseUncertainty, GaussianUncertainty


logger = logging.getLogger(__name__)


class PositionalGridSurvey(skysurvey.GridSurvey):

    def __init__(self, survey, uncertainty: BaseUncertainty):
        super().__init__(survey)
        self.uncertainty = uncertainty

    def draw_positions(self, n: int, truth: tuple[float, float]) -> list[tuple[float, float]]:
        return self.uncertainty.draw_position(n, truth)
