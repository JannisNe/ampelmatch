import logging
from pathlib import Path

import requests
from astropy.table import Table

from ampelmatch.cache import cache_dir

logger = logging.getLogger(__name__)


class Fermi4LAC:
    cache_file = Path(cache_dir) / "fermi_4lac_dr2.fits"
    selection_file = Path(cache_dir) / "fermi_4lac_dr2_selection.csv"

    def __init__(self):
        self.data = self.load_data()

    @staticmethod
    def load_data():
        if not Fermi4LAC.cache_file.exists():
            Fermi4LAC.fetch_data()
        return Table.read(Fermi4LAC.cache_file)

    @staticmethod
    def fetch_data():
        url = "https://www.ssdc.asi.it/fermi4lac-DR2/table-4LAC-DR2-h.fits"
        logger.info("Fetching Fermi 4LAC DR2 data")
        r = requests.get(url)
        r.raise_for_status()
        with Fermi4LAC.cache_file.open("wb") as f:
            f.write(r.content)
        logger.info(f"Saved Fermi 4LAC DR2 data to {Fermi4LAC.cache_file}")

    def make_selection(self):
        data = self.data.to_pandas()
        classm = data.CLASS.str.decode("utf-8").isin(
            ["BCU", "BLL", "FSRQ", "bll", "bcu", "fsrq"]
        )
        fluxm = data.Energy_Flux100 > (10 ** (-11.6))
        m = classm & fluxm
        logger.info(f"Selected {m.sum()} sources")
        data[classm & fluxm].to_csv(Fermi4LAC.selection_file, index=False)
        logger.info(f"Saved selection to {Fermi4LAC.selection_file}")
