import ampelmatch
import logging
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class BaseUncertaintyConfig(BaseModel):
    """
    Configuration class for uncertainties
    """
    type: str


class GaussianUncertaintyConfig(BaseUncertaintyConfig):
    """
    Configuration class for Gaussian uncertainties
    """
    sigma_arcsec: float


class BasePositionalSurveyConfig(BaseModel):
    """
    Configuration class for surveys
    """
    name: str
    gain: float
    zp: float
    skynoise_mean: float
    time_min: str
    time_max: str
    bands: list
    size: int
    uncertainty: BaseUncertaintyConfig


class PositionalGridSurveyConfig(BasePositionalSurveyConfig):
    """
    Configuration class for grid surveys
    """
    fields: dict
    fov: float


class TransientConfig(BaseModel):
    """
    Configuration class for transients
    """
    name: str
    draw: int
    zmax: float


class DatasetConfig(BaseModel):
    """
    Configuration class for ampelmatch
    """
    name: str
    surveys: list[BasePositionalSurveyConfig]
    transients: list[TransientConfig]
