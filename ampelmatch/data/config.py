from typing import Literal, Annotated, Union
import logging
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class BaseUncertaintyConfig(BaseModel):
    """
    Configuration class for uncertainties
    """
    uncertainty_type: str


class GaussianUncertaintyConfig(BaseUncertaintyConfig):
    """
    Configuration class for Gaussian uncertainties
    """
    uncertainty_type: Literal["GaussianUncertainty"]
    sigma_arcsec: float


Uncertainty = Annotated[Union[GaussianUncertaintyConfig], Field(..., discriminator="uncertainty_type")]


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
    uncertainty: Uncertainty


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
    transient_type: str
    draw: int
    zmax: float
    tstart: str
    tstop: str


class SNIaConfig(TransientConfig):
    """
    Configuration class for SNIa transients
    """
    transient_type: Literal["SNIa"]


Transient = Annotated[Union[SNIaConfig], Field(..., discriminator="transient_type")]


class DatasetConfig(BaseModel):
    """
    Configuration class for ampelmatch
    """
    name: str
    surveys: list[Survey]
    transients: list[Transient]
