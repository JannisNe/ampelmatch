import logging
import skysurvey
from cachier import cachier

from ampelmatch.cache import cache_dir, model_hash
from ampelmatch.data.config import Transient


logger = logging.getLogger(__name__)


class TransientGenerator:
    transient_classes = {
        "SNIa": skysurvey.SNeIa
    }

    def __init__(self, configs: list[Transient]):
        self.configs = configs
        self.iter_config = iter(configs)

    @staticmethod
    @cachier(cache_dir=cache_dir, hash_func=model_hash)
    def realize_transient_data(config):
        logger.info(f"Generating {config}")
        transient = TransientGenerator.transient_classes[config.transient_type]()
        transient.draw(
            size=config.size,
            tstart=config.tstart,
            tstop=config.tstop,
            zmax=config.zmax,
            skyarea=config.skyarea,
            inplace=True
        )
        logger.info(f"Generated {len(transient.data)} {config.transient_type} transients")
        return transient.data

    def __iter__(self):
        logger.info("Making test transients")
        return self

    def __next__(self):
        config = next(self.iter_config)
        transient = self.transient_classes[config.transient_type]()
        transient.set_data(self.realize_transient_data(config))
        return transient
