####################################################################################################
# sco/normalization/core.py
# Second-order-contrast normalization and nonlinearity application
# By Noah C. Benson

import numpy as     np
from ..core  import calculates


@calculates()
def calc_normalization_default_parameters(pRF_v123_labels, Kay2013_response_gain=1,
                                          Kay2013_output_nonlinearity={1: 0.18, 2: 0.13, 3: 0.12},
                                          Kay2013_SOC_constant={1: 0.93, 2: 0.99, 3: 0.99}):
    """
    calc_normalization_default_parameters() is a calculator that expects no particular options,
    but fills in several options if not provided: 
      * Kay2013_response_gain
      * Kay2013_output_nonlinearity
      * Kay2013_SOC_constant
      
    For all of these, they can be a single float, a list/array of float, or a dictionary with 1, 2
    and 3 as its keys and with floats as the values, specifying the values for these parameters for
    voxels in areas V1, V2, and V3. This function will take these values and form arrays that
    correspond to the other voxel-level arrays.
    """
    return {'Kay2013_response_gain':       _turn_param_into_array(Kay2013_response_gain,
                                                                  pRF_v123_labels),
            'Kay2013_output_nonlinearity': _turn_param_into_array(Kay2013_output_nonlinearity,
                                                                  pRF_v123_labels),
            'Kay2013_SOC_constant':        _turn_param_into_array(Kay2013_SOC_constant,
                                                                  pRF_v123_labels)}

@calculates('SOC_responses')
def calc_Kay2013_SOC_normalization(pRF_views,
                                   normalized_pixels_per_degree,
                                   normalized_contrast_functions,
                                   pRF_frequency_preferences,
                                   Kay2013_SOC_constant):
    """
    Calculate the second-order contrast
    """
    # we can use the (PRFSpec class) pRF_view elements' __call__ function to do this calculation
    # efficiently; see the pRF/core.py source code for more information, or the sco.pRF.PRFSpec
    # class documentation.
    return {'SOC_responses': np.asarray(
        [ [ view(im, d2p, c=cval)
            for (ncf, d2p) in zip(ncfs, normalized_pixels_per_degree)
            for im         in [np.sum([v*ncf(k) for (k,v) in prefs.iteritems()], axis=0)]]
          for (view, ncfs, cval, prefs) in zip(pRF_views,
                                               normalized_contrast_functions,
                                               Kay2013_SOC_constant,
                                               pRF_frequency_preferences)])}
    ## Because pRF_matrices are all sparse and sum to 1, we can use them cleverly to save time:
    #responses = np.zeros(pRF_matrices.shape)
    #for (i, (cfns, ws, prefs, c)) in enumerate(zip(normalized_contrast_functions,
    #                                               pRF_matrices,
    #                                               pRF_frequency_preferences,
    #                                               Kay2013_SOC_constant)):
    #    for (j, (cfn, w)) in enumerate(zip(cfns, ws)):
    #        # make the relevant image...
    #        im = np.sum([cfn(k) * v for (k,v) in prefs.iteritems()], axis=0)
    #        # get the mean value of the pRF
    #        mu = w.multiply(im).sum()
    #        # find the pixels we care about and subtract this from them
    #        w1 = w.astype(bool)
    #        blob_mu = w1 * (c * mu)
    #        blob = w1.multiply(im) - blob_mu
    #        # calculate the response
    #        responses[i,j] = w.multiply(blob.multiply(blob)).sum()
    #return {'SOC_responses': responses}
    
@calculates('predicted_responses')
def calc_Kay2013_output_nonlinearity(SOC_responses, Kay2013_output_nonlinearity,
                                     Kay2013_response_gain):
    """Calculate the compressive nonlinearity

    This is the final step of the model; it's output is the predicted
    response.
    """
    out = np.asarray([gval * q ** nval for (q, nval, gval) in zip(SOC_responses,
                                                                  Kay2013_output_nonlinearity,
                                                                  Kay2013_response_gain)]).T
    return {'predicted_responses': out}

def _turn_param_into_array(param, pRF_v123_labels):
    """takes param and turns it into an array

    We have a handful of parameters that can be:

    - an integer, in which case we want to use the same value for each voxel

    - a dictionary, in which case we want to each voxel to use the value corresponding to its
    pRF_v123_label

    - a list/array, in which case we assume that there's already one value for each voxel and we
    cast it as an array and use that.
    """
    if not hasattr(param, '__iter__'):
        # then vals is a single value and the same for everything
        vals = np.repeat(float(param), len(pRF_v123_labels))
    else:
        try:
            # if this succeeds, it's a dictionary
            vals = np.asarray([param.get(l) for l in pRF_v123_labels])
        except AttributeError:
            # then it's a list or array and we cast it as an array
            vals = np.asarray(param)
    return vals
