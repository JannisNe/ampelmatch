import logging
import skysurvey

from ampelmatch.data.positional_uncertainty import BaseUncertainty, GaussianUncertainty


logger = logging.getLogger(__name__)


class PositionalGridSurvey(skysurvey.GridSurvey):

    def __init__(self, uncertainty: BaseUncertainty, data=None, fields=None):
        super().__init__(data, fields)
        self.uncertainty = uncertainty

    @classmethod
    def from_pointings(cls, data, fields_or_coords=None, footprint=None, uncertainty: BaseUncertainty = None, **kwargs):
        _survey = skysurvey.GridSurvey.from_pointings(data, fields_or_coords, footprint, **kwargs)
        _survey.uncertainty = uncertainty
        return _survey

    def draw_positions(self, n: int, truth: tuple[float, float]) -> list[tuple[float, float]]:
        return self.uncertainty.draw_position(n, truth)
