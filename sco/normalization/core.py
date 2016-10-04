####################################################################################################
# sco/normalization/core.py
# Second-order-contrast normalization and nonlinearity application
# By Noah C. Benson

import numpy                 as     np
import scipy                 as     sp

from   numbers               import Number
from   pysistence            import make_dict

import os, math, itertools, collections, abc

from ..core import (calculates, calc_chain, iscalc)


@calculates('SOC_normalized_responses')
def calc_Kay2013_SOC_normalization(pRF_responses, pRF_v123_labels,
                                   Kay2013_SOC_constant={1: 0.93, 2: 0.99, 3: 0.99}):
    """Calculate the second-order contrast
    """
    c = 1.0 - np.asarray([Kay2013_SOC_constant[l] for l in pRF_v123_labels])
    normed = np.asarray([(np.asarray(r) * cval)**2 for (r,cval) in zip(pRF_responses, c)])
    return {'SOC_normalized_responses': normed}


@calculates('predicted_responses')
def calc_Kay2013_output_nonlinearity(SOC_normalized_responses, pRF_v123_labels,
                                     Kay2013_output_nonlinearity):
    """Calculate the compressive nonlinearity

    This is the final step of the model; it's output is the predicted
    response.
    """
    n = np.asarray([Kay2013_output_nonlinearity[l] for l in pRF_v123_labels])
    out = np.asarray([q ** nval for (q,nval) in zip(SOC_normalized_responses, n)]).T
    return {'predicted_responses': out}
