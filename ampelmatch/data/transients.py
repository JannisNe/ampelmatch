import logging
import skysurvey
import diskcache

from ampelmatch import cache_dir
from ampelmatch.data.config import TransientConfig


logger = logging.getLogger(__name__)
cache = diskcache.Cache(cache_dir)


class TransientGenerator:
    transient_classes = {
        "SNIa": skysurvey.SNeIa
    }

    def __init__(self, configs: list[TransientConfig]):
        self.configs = configs
        self.iter_config = iter(configs)

    @staticmethod
    @cache.memoize()
    def realize_transient_data(config):
        transient = TransientGenerator.transient_classes[config.name]()
        transient.draw(
            tstart=config.tstart,
            tstop=config.tstop,
            zmax=config.zmax,
            inplace=True
        )
        logger.info(f"Generated {len(transient.data)} {config.name} transients")
        return transient.data

    def __iter__(self):
        logger.debug("Generating test transients")
        return self

    def __next__(self):
        config = next(self.iter_config)
        transient = self.transient_classes[config.name]()
        transient.set_data(self.realize_transient_data(config))
        return transient
