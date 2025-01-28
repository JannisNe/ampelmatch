import logging
from pathlib import Path
import shapely
import typer

from ampelmatch.data.config import DatasetConfig, Transient
from ampelmatch.data.surveys import SurveyGenerator


logger = logging.getLogger("ampelmatch.data.extract_survey_area")


def rewrite_config_json(filename: str, logging_level: str = "INFO", buffer: float = 0.0):
    logging.getLogger("ampelmatch").setLevel(logging_level)
    filename = Path(filename)
    logger.info(f"Extracting survey area from {filename}")
    dset_config = DatasetConfig.model_validate_json(filename.read_text())
    shape = shapely.union_all([s.get_skyarea() for s in SurveyGenerator(dset_config.surveys)])
    for tc in dset_config.transients:
        tc.skyarea = shape.buffer(buffer)
    filename.write_text(dset_config.model_dump_json(indent=4))
    logger.info(f"Saved survey area to {filename}")


if __name__ == "__main__":
    typer.run(rewrite_config_json)
