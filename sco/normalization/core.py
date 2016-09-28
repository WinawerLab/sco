##########################################################################
# sco/normalization/core.py
# Second-order-contrast normalization and nonlinearity application
# By Noah C. Benson

import numpy as np
import scipy as sp

##########################################################################
# normalization/core.py
#
# The normalization module of the standard cortical observer; core
# definitions and checks.
#
# By Noah C. Benson

from ..core import calculates


@calculates('SOC_normalized_responses')
def calc_Kay2013_SOC_normalization(pRF_responses, pRF_v123_labels,
                                   Kay2013_SOC_constant={1: 0.93, 2: 0.99,
                                                         3: 0.99}):
    """Calculate the second-order contrast
    """
    c = 1.0 - np.asarray([Kay2013_SOC_constant[l] for l in pRF_v123_labels])
    normed = [(np.asarray(r) * c)**2 for r in pRF_responses]
    return {'SOC_normalized_responses': normed}


@calculates('predicted_responses')
def calc_Kay2013_output_nonlinearity(SOC_normalized_responses, pRF_v123_labels,
                                     Kay2013_output_nonlinearity):
    """Calculate the compressive nonlinearity

    This is the final step of the model; it's output is the predicted
    response.
    """
    n = np.asarray([Kay2013_output_nonlinearity[l] for l in pRF_v123_labels])
    out = SOC_normalized_responses ** n
    return {'predicted_responses': out}
