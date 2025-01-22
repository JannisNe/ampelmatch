import logging
import numpy as np
import pandas as pd
import skysurvey
import hashlib
import json
from pathlib import Path
from shapely import union_all

from ampelmatch.data.surveys import survey_params, generate_test_surveys


logger = logging.getLogger(__name__)


zmax = 1
transient_config = {
    "SNeIa": {
        "draw": 10_000,
        "zmax": zmax
    }
}


def get_surveys_info():
    skyareas = []
    time_ranges = []
    for s in generate_test_surveys():
        skyareas.append(s.get_skyarea())
        time_ranges.append(s.get_timerange())
    time_ranges = np.array(time_ranges)
    t_start = np.min(time_ranges[:, 0])
    t_stop = np.max(time_ranges[:, 1])
    skyarea = union_all(skyareas)
    return t_start, t_stop, skyarea


def get_transient_hash(transient_name: str):
    d = transient_config[transient_name].update(survey_params)
    return hashlib.md5(json.dumps(d).encode()).hexdigest()


def generate_transient_sample():
    for transient_name, config in transient_config.items():
        fname = Path(f"{transient_name}_{get_transient_hash(transient_name)}.csv")
        if not fname.exists():
            logger.info(f"Generating {config['draw']} {transient_name} transients")
            tstart, tstop, skyarea = get_surveys_info()
            t = skysurvey.__getattribute__(transient_name).from_draw(
                tstart=tstart, tstop=tstop,
                skyarea=skyarea,
                zmax=config['zmax']
            )
            t.save(fname)
            logger.info(f"Saved {config['draw']} {transient_name} transients to {fname}")
        yield transient_name, pd.read_csv(fname, index_col=0)