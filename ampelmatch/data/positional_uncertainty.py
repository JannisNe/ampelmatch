import ampelmatch
import logging
import skysurvey
from astropy.coordinates import SkyCoord


logger = logging.getLogger(__name__)


class BaseUncertainty:
    def draw_position(self, n: int, truth: tuple[float, float]) -> list[tuple[float, float]]:
        ...


class GaussianUncertainty(BaseUncertainty):
    def __init__(self, sigma_arcsec: float):
        self.sigma = sigma_arcsec

    def draw_position(self, n: int, truth: tuple[float, float]) -> list[tuple[float, float]]:
        coords = SkyCoord(*truth, unit='deg')
        offsets = np.random.normal(0, self.sigma, n)
        ps = np.random.uniform(0, 2 * np.pi, n)
        new_coords = coords.directional_offset_by(ps, offsets * u.arcsec)
        return [(c.ra.deg, c.dec.deg) for c in new_coords]