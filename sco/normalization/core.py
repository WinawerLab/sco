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

Kay2013_pRF_sigma_slope     = {1: 0.1,  2: 0.15, 3: 0.27}
Kay2013_SOC_constant        = {1: 0.93, 2: 0.99, 3: 0.99}
Kay2013_output_nonlinearity = {1: 0.18, 2: 0.13, 3: 0.12}

@calculates('SOC_normalized_responses')
def calc_Kay2013_SOC_normalization(pRF_responses, pRF_v123_labels):
    c = 1.0 - np.asarray([Kay2013_SOC_constant[l] for l in pRF_v123_labels])
    normed = [(np.asarray(r) * c)**2 for r in pRF_responses]
    return {'SOC_normalized_responses': normed}

@calculates('predicted_responses')
def calc_Kay2013_output_nonlinearity(SOC_normalized_responses, pRF_v123_labels):
    n = np.asarray([Kay2013_output_nonlinearity[l] for l in pRF_v123_labels])
    out = SOC_normalized_responses ** n
    return {'predicted_responses': out}

    
