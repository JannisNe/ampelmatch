import logging
import hashlib
import json
from pathlib import Path
import skysurvey
import pandas as pd

from ampelmatch.data.positional_dataset import PositionalDataset
from ampelmatch.data.transients import transient_config, get_transient_hash, generate_transient_sample
from ampelmatch.data.surveys import survey_params, get_survey_hash, generate_test_surveys


logger = logging.getLogger(__name__)


def get_hash():
    d = transient_config.update(survey_params)
    return hashlib.md5(json.dumps(d).encode()).hexdigest()


def generate_transient_data(survey_name: str, survey: skysurvey.Survey,  transient_name: str, targets: skysurvey.Target):
    survey_hash = get_survey_hash(survey_name)
    transient_hash = get_transient_hash(transient_name)
    fname = Path(f"{survey_name}_{survey_hash}_{transient_name}_{transient_hash}.csv")
    if not fname.exists():
        logger.info(f"Generating {transient_name} transients for {survey_name}")
        PositionalDataset.from_targets_and_survey(targets, survey).data.to_csv(fname)
        logger.info(f"Saved {transient_name} transients for {survey_name} to {fname}")
    return pd.read_csv(fname, index_col=0)


def generate_test_data():
    for sname, survey in generate_test_surveys():
        dfs = []
        for tname, t in generate_transient_sample():
            dfs.append(generate_transient_data(sname, survey, tname, t))
        yield sname, pd.concat(dfs)
