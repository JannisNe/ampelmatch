import importlib
import logging
from pathlib import Path

from ampelmatch.data.fermi_4lac_dr2 import Fermi4LAC
from ampelmatch.data.icecube_alert import IceCubeAlerts
from ampelmatch.match import match

match = importlib.reload(match)

logger = logging.getLogger("ampelmatch.match.test_gaussian")


if __name__ == "__main__":
    logging.getLogger("ampelmatch").setLevel("DEBUG")
    fermi_4lac = Fermi4LAC()
    if not fermi_4lac.selection_file.is_file():
        fermi_4lac.make_selection()

    icecube_alerts = IceCubeAlerts()
    icecube_alert_filename = Path("icecube_alerts.csv")
    if not icecube_alert_filename.is_file():
        icecube_alerts.load_data()
        icecube_alerts.write_data(icecube_alert_filename)

    match_config = {
        "bayes_factor": {
            "name": f"fermi_4lac_test",
            "match_type": "gaussian",
            "plot": 10,
            "nside": 1024,
            "primary_data": {
                "filepath_or_buffer": fermi_4lac.selection_file,
                "index_col": 0,
                # "nrows": 700,
            },
            "match_data": [
                {
                    "filepath_or_buffer": fn,
                }
                for fn in fns[1:]
            ],
        },
        "prior": {
            "name": "surface_density",
            "nside": 128,
            "area_sqdg": a,
            "primary_data": {
                "filepath_or_buffer": fns[0],
                "index_col": 0,
                # "nrows": 700,
            },
            "match_data": [
                {
                    "filepath_or_buffer": fermi_4lac.selection_file,
                }
            ],
        },
        "posterior_threshold": 0.95,
    }
    matcher = match.StreamMatch.model_validate(match_config)
    probabilities = matcher.posteriors
    match = matcher.match()
