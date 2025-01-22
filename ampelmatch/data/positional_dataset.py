import ampelmatch
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
        assert isinstance(survey, BaseUncertainty), "Survey must have uncertainties"
        lc_out, fieldids_indexes = super()._realize_survey_kindtarget_lcs()
        # TODO: implement drawing positions and errors
        return lc_out, fieldids_indexes
