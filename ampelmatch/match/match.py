import logging
from typing import Annotated

from cryptography.utils import cached_property
from pydantic import BaseModel, Field
from tqdm import tqdm

from ampelmatch.match.bayes_factor import BayesFactor
from ampelmatch.match.prior import Prior

logger = logging.getLogger(__name__)


class StreamMatch(BaseModel):
    bayes_factor: Annotated[BayesFactor, Field(discriminator="match_type")]
    prior: Annotated[Prior, Field(discriminator="name")]
    posterior_threshold: float

    @cached_property
    def posteriors(self):
        logger.info("Calculating probabilities")
        bayes_factors = self.bayes_factor.evaluate()
        posteriors = {}
        self.bayes_factor.plot_dir = self.bayes_factor.plot_dir / "posteriors"
        for source_id, bf in tqdm(
            bayes_factors.items(),
            desc="Calculating posteriors",
            total=len(bayes_factors),
        ):
            i_posteriors = {}

            if source_id in self.bayes_factor.plot_indices:
                primary_data = self.bayes_factor.primary_data_df.loc[source_id]
                fig, ax, axs = self.bayes_factor.setup_plot(primary_data, len(bf))

            for sd_id, sdbf in bf.items():
                p = self.prior(self.bayes_factor.primary_data_df.loc[source_id])
                i_posteriors[sd_id] = (1 + (1 - p) / (p * sdbf)) ** (-1)

                if source_id in self.bayes_factor.plot_indices:
                    orig_sources = self.bayes_factor.match_data_df[sd_id].loc[
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
                        cbar_lim=(0, 1),
                    )

            if source_id in self.bayes_factor.plot_indices:
                self.bayes_factor.finalize_plot(fig, ax, source_id)

            posteriors[source_id] = i_posteriors
        return posteriors

    def match(self) -> dict:
        return {
            source_id: {
                sd_id: post.index[post > self.posterior_threshold].tolist()
                for sd_id, post in posts.items()
            }
            for source_id, posts in self.posteriors.items()
        }

    def n_matches(self) -> list[float]:
        return [
            sum([len(v[i]) for v in self.match().values()])
            for i in range(len(self.bayes_factor.match_data_df))
        ]

    def posterior_sum(self) -> list[float]:
        return [
            sum([v[i].sum() for v in self.posteriors.values()])
            for i in range(len(self.bayes_factor.match_data_df))
        ]
