import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import requests
import tarfile
from pathlib import Path
from typing import Iterator
import healpy as hp

from ampelmatch.cache import cache_dir


logger = logging.getLogger(__name__)


class IceCubeAlerts:

    DATAVERSE_IDS = {
        2011: 6933693,
        2012: 6933690,
        2013: 6933688,
        2014: 6933686,
        2015: 6933689,
        2016: 6933692,
        2017: 6933694,
        2018: 6933691,
        2019: 6933695,
        2020: 6933687,
        2021: 7502708,
        2022: 7502709,
        2023: 7502707
    }

    def __init__(self):
        self.data = None
        self.load_data()

    def load_data(self):
        files = []
        for year, dataverse_id in self.DATAVERSE_IDS.items():
            logger.info(f"Downloading IceCube alerts for {year}")
            files.extend(list(self.get_icecube_alerts(dataverse_id)))

        _, h = hp.read_map(files[0], h=True)
        dtypes = [
            (k, type(v)) for k, v in dict(h).items()
        ] + [("map", object), ("filename", str)]
        data = np.array(np.empty((len(files), len(dtypes))), dtype=dtypes)
        for i, f in enumerate(files):
            logger.info(f"Reading {f}")
            s, h = hp.read_map(f, h=True)
            i_data = dict(h)
            i_data["map"] = s
            i_data["filename"] = f
            data[i] = tuple(i_data.values())

        self.data = pd.DataFrame(data)

    def write_data(self, filename: str | Path):
        filename = Path(filename)
        logger.info(f"Writing IceCube alerts to {filename}")
        self.data.to_csv(filename, index=False)

    @staticmethod
    def get_icecube_alerts(dataverse_id: str) -> Iterator[Path]:

        tar_dir = Path(cache_dir) / dataverse_id

        if not tar_dir.exists():
            tar_dir.mkdir(exist_ok=True, parents=True)
            cache_file = tar_dir.with_suffix(".tar")
            url = f"https://dataverse.harvard.edu/api/access/datafile/{dataverse_id}"
            with requests.get(url, stream=True) as r:
                with cache_file.open("wb") as f:
                    for chunk in tqdm(r.iter_content(chunk_size=8192)):
                        f.write(chunk)

            with tarfile.open(cache_file, "r") as tar:
                tar.extractall(tar_dir)

            cache_file.unlink()

        return tar_dir.glob("*.fits.gz")