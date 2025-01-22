import logging
import skysurvey

from ampelmatch.data.positional_uncertainty import BaseUncertainty, GaussianUncertainty


logger = logging.getLogger(__name__)


class PositionalGridSurvey(skysurvey.GridSurvey):

    def __init__(self, uncertainty: BaseUncertainty, data=None, fields=None):
        super().__init__(data, fields)
        self.uncertainty = uncertainty

    @classmethod
    def from_pointings(cls, data, footprint=None, moc=None, rakey="ra", deckey="dec", uncertainty: BaseUncertainty = None):
        _survey = super().from_pointings(data, footprint, moc, rakey, deckey)
        _survey.uncertainty = uncertainty
        return _survey

    def draw_positions(self, n: int, truth: tuple[float, float]) -> list[tuple[float, float]]:
        return self.uncertainty.draw_position(n, truth)
