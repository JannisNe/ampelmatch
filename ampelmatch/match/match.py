import logging
from typing import Annotated

from ampelmatch.match.bayes_factor import BayesFactor
from ampelmatch.match.prior import Prior
from pydantic import BaseModel, Field
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StreamMatch(BaseModel):
    bayes_factor: Annotated[BayesFactor, Field(discriminator="match_type")]
    prior: Annotated[Prior, Field(discriminator="name")]

    def calculate_posteriors(self):
        logger.info("Calculating probabilities")
        bayes_factors = self.bayes_factor.match()
        posteriors = {}
        self.bayes_factor.plot_dir = self.bayes_factor.plot_dir / "posteriors"
        for source_id, bf in tqdm(
            bayes_factors.items(),
            desc="Calculating posteriors",
            total=len(bayes_factors),
        ):
            i_posteriors = {}

            if source_id in self.bayes_factor.plot_indices:
                primary_data = self.bayes_factor._primary_data.loc[source_id]
                fig, ax, axs = self.bayes_factor.setup_plot(primary_data, len(bf))

            for sd_id, sdbf in bf.items():
                p = self.prior.evaluate_prior(
                    self.bayes_factor._primary_data.loc[source_id],
                    [self.bayes_factor._match_data[sd_id]],
                )
                i_posteriors[sd_id] = sdbf * p / (sdbf * p + 1)

                if source_id in self.bayes_factor.plot_indices:
                    orig_sources = self.bayes_factor._match_data[sd_id].loc[
                        i_posteriors[sd_id].index
                    ]
                    orig_sources.loc[:, "marker"] = "s"
                    orig_sources.loc[:, "post"] = i_posteriors[sd_id]
                    self.bayes_factor.add_data_to_plot(
                        ax,
                        orig_sources,
                        "post",
                        self.bayes_factor.cmaps[sd_id],
                        f"posterior {sd_id}",
                        axs[sd_id],
                    )

            if source_id in self.bayes_factor.plot_indices:
                self.bayes_factor.finalize_plot(fig, ax, source_id)

            posteriors[source_id] = i_posteriors
        return posteriors
