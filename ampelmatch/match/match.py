import logging
from typing import Annotated

from pydantic import BaseModel, Field

from ampelmatch.match.bayes_factor import BayesFactor
from ampelmatch.match.prior import Prior

logger = logging.getLogger(__name__)


class StreamMatch(BaseModel):
    bayes_factor: Annotated[BayesFactor, Field(discriminator="match_type")]
    prior: Annotated[Prior, Field(discriminator="name")]

    def calculate_posteriors(self):
        logger.info("Calculating probabilities")
        bayes_factors = self.bayes_factor.match()
        posteriors = {}
        for source_id, bf in bayes_factors.items():
            i_posteriors = {}
            for sd_id, sdbf in bf.items():
                p = self.prior.evaluate_prior(
                    self.bayes_factor._primary_data.loc[source_id],
                    [self.bayes_factor._match_data[sd_id]],
                )
                i_posteriors[sd_id] = sdbf * p / (sdbf * p + 1)
            posteriors[source_id] = i_posteriors
        return posteriors
