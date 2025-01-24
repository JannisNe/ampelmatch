from typing import Literal, Annotated, Union
import logging
from pydantic import BaseModel, field_validator, Field


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
    survey_type: str
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
    survey_type: Literal["PositionalGridSurvey"]
    fields: dict
    fov: float


Survey = Annotated[Union[PositionalGridSurveyConfig], Field(..., discriminator="survey_type")]


class TransientConfig(BaseModel):
    """
    Configuration class for transients
    """
    name: str
    draw: int
    zmax: float
    tstart: str
    tstop: str


class DatasetConfig(BaseModel):
    """
    Configuration class for ampelmatch
    """
    name: str
    surveys: list[Survey]
    transients: list[TransientConfig]
