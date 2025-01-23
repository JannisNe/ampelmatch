import numpy as np
import logging
import skysurvey

from ampelmatch.data.positional_uncertainty import BaseUncertainty


logger = logging.getLogger(__name__)


class PositionalDataset(skysurvey.DataSet):

    @staticmethod
    def _realize_survey_kindtarget_lcs( targets, survey, template=None,
                                           template_prop={}, nfirst=None,
                                           incl_error=True,
                                           client=None, discard_bands=False):
        assert hasattr(survey, 'uncertainty'), "Survey must have uncertainties"
        unc = survey.uncertainty
        assert isinstance(unc, BaseUncertainty), "uncertainty must be a subclass of BaseUncertainty"
        lc_out, fieldids_indexes = super(PositionalDataset, PositionalDataset)._realize_survey_kindtarget_lcs(
            targets, survey, template, template_prop, nfirst, incl_error, client, discard_bands
        )
        for c in unc.POSITION_KEYS + unc.PARAMETER_KEYS:
            lc_out[0][c] = np.nan

        for ilc, ifid in zip(lc_out, fieldids_indexes):
            for target_index in ilc.index.levels[0]:
                iilc = ilc.loc[target_index]
                parameters = unc.draw_position(iilc, truth=targets.data.loc[target_index])
                for c, v in zip(unc.POSITION_KEYS + unc.PARAMETER_KEYS, parameters):
                    ilc.loc[target_index, c] = v

        return lc_out, fieldids_indexes
