import ampelmatch
import logging
import skysurvey
from astropy.coordinates import SkyCoord


logger = logging.getLogger(__name__)


class BaseUncertainty:
    registry = {}

    def draw_position(self, n: int, truth: tuple[float, float]) -> list[tuple[float, float]]:
        ...

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ in BaseUncertainty.registry:
            raise ValueError(f"Duplicate uncertainty class name: {cls.__name__}")
        BaseUncertainty.registry[cls.__name__] = cls

    @classmethod
    def from_dict(cls, data: dict) -> 'BaseUncertainty':
        if "type" not in data:
            raise ValueError("Uncertainty type not provided")
        _type = data.pop("type")
        if _type not in BaseUncertainty.registry:
            raise ValueError(f"Uncertainty type not recognized: {_type}")
        return BaseUncertainty.registry[_type](**data)


class GaussianUncertainty(BaseUncertainty):
    def __init__(self, sigma_arcsec: float):
        self.sigma = sigma_arcsec

    def draw_position(self, n: int, truth: tuple[float, float]) -> list[tuple[float, float]]:
        coords = SkyCoord(*truth, unit='deg')
        offsets = np.random.normal(0, self.sigma, n)
        ps = np.random.uniform(0, 2 * np.pi, n)
        new_coords = coords.directional_offset_by(ps, offsets * u.arcsec)
        return [(c.ra.deg, c.dec.deg) for c in new_coords]