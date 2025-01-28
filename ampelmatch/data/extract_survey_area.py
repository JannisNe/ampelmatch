import logging
from pathlib import Path
import shapely
import typer

from ampelmatch.data.config import DatasetConfig
from ampelmatch.data.surveys import SurveyGenerator


logger = logging.getLogger("ampelmatch.data.extract_survey_area")


def extract_survey_area(filename: str, logging_level: str = "INFO", buffer: float = 0.0):
    logging.getLogger("ampelmatch").setLevel(logging_level)
    filename = Path(filename)
    logger.info(f"Extracting survey area from {filename}")
    dset_config = DatasetConfig.model_validate_json(filename.read_text())
    shape = shapely.union_all([s.get_skyarea() for s in SurveyGenerator(dset_config.surveys)])
    out_filename = filename.parent / f"{dset_config.name}_survey_area.json"
    out_filename.write_text(shapely.to_geojson(shape.buffer(buffer), indent=4))
    logger.info(f"Saved survey area to {out_filename}")


if __name__ == "__main__":
    typer.run(extract_survey_area)
