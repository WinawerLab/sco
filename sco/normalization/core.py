####################################################################################################
# sco/normalization/core.py
# Second-order-contrast normalization and nonlinearity application
# By Noah C. Benson

import numpy as     np
from ..core  import calculates



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
                                     Kay2013_output_nonlinearity, Kay2013_response_gain=1):
    """Calculate the compressive nonlinearity

    This is the final step of the model; it's output is the predicted
    response.
    """
    n = _turn_param_into_list(Kay2013_output_nonlinearity, pRF_v123_labels)
    g = _turn_param_into_list(Kay2013_response_gain, pRF_v123_labels)
    out = np.asarray([gval * q ** nval for (q, nval, gval) in zip(SOC_normalized_responses, n, g)]).T
    return {'predicted_responses': out}

def _turn_param_into_list(param, pRF_v123_labels):
    """takes param and turns it into a list

    We have a handful of parameters that can be:

    - an integer, in which case we want to use the same value for each voxel

    - a dictionary, in which case we want to each voxel to use the value corresponding to its
    pRF_v123_label

    - a list/array, in which case we assume that there's already one value for each voxel and we
    use it as is
    """
    if not hasattr(param, '__iter__'):
        # then vals is a single value and the same for everything
        vals = np.repeat(param, len(pRF_v123_labels))
    else:
        try:
            # if this succeeds, it's a dictionary
            vals = np.asarray([param.get(l) for l in pRF_v123_labels])
        except AttributeError:
            # then it's a list and we can use it as is.
            vals = param
    return vals
