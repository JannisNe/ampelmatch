import importlib
import logging
from pathlib import Path

from ampelmatch.data.fermi_4lac_dr2 import Fermi4LAC
from ampelmatch.data.icecube_alert import IceCubeAlerts
from ampelmatch.match import match

match = importlib.reload(match)

logger = logging.getLogger("ampelmatch.match.test_gaussian")


if __name__ == "__main__":
    logging.getLogger("ampelmatch").setLevel("INFO")
    fermi_4lac = Fermi4LAC()
    fermi_4lac.make_selection()
    if not fermi_4lac.selection_file.is_file():
        fermi_4lac.dump_selection()

    icecube_alerts = IceCubeAlerts()
    icecube_alert_filename = Path("icecube_alerts.csv")
    if not icecube_alert_filename.is_file():
        icecube_alerts.load_data()
        icecube_alerts.write_data(icecube_alert_filename)

    match_config = {
        "bayes_factor": {
            "name": f"fermi_4lac_test",
            "match_type": "icecube_contour",
            "plot": False,
            "nside": 1024,
        },
        "prior": {
            "name": "ra_scramble",
            "bayes_factor": {
                "name": f"fermi_4lac_test_prior",
                "match_type": "icecube_contour",
                "plot": False,
                "nside": 1024,
            },
            "n_scrambles": 100,
            "primary_data": {
                "filepath_or_buffer": fermi_4lac.selection_file,
            },
            "match_data": [
                {
                    "filepath_or_buffer": icecube_alert_filename,
                }
            ],
        },
        "posterior_threshold": 0.95,
        "primary_data": {
            "filepath_or_buffer": fermi_4lac.selection_file,
        },
        "match_data": [
            {
                "filepath_or_buffer": icecube_alert_filename,
            }
        ],
    }
    matcher = match.StreamMatch.model_validate(match_config)
    scrambles = matcher.prior.scrambled_bayes_factors()
    w = (
        fermi_4lac.selection["Energy_Flux100"]
        / fermi_4lac.selection["Energy_Flux100"].sum()
    )
