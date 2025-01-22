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
        assert hasattr(survey, 'uncertainty'), "Survey must have uncertainties"
        unc = survey.uncertainty
        assert isinstance(unc, BaseUncertainty), "uncertainty must be a subclass of BaseUncertainty"
        lc_out, fieldids_indexes = skysurvey.DataSet._realize_survey_kindtarget_lcs(
            targets, survey, template, template_prop, nfirst, incl_error, client, discard_bands
        )
        # TODO: implement drawing positions and errors
        return lc_out, fieldids_indexes
