####################################################################################################
# sco/impl/benson17/__init__.py
# Implementation details of the SCO specific to the Benson et al. 2017 publications.
# by Noah C. Benson

'''
The sco.impl.benson17 namespace contains default values for all of the optional parameters used in
the Winawer lab implementation of the SCO model. A function, provide_default_options is also
defined; this may be included in any SCO plan to ensure that the optional parameter values are
available to the downstream calculations if not provided by the user or other calculations.
'''

import pyrsistent as _pyr
import pimms      as _pimms
import numpy      as _np

# Divisive Normalization ###########################################################################
# The following are divisive normalization models that can be used with the SCO
def divisively_normalize_Heeger1992(data, response_exponent=0.5, saturation_constant=1.0):
    '''
    divisively_normalize(data) yields the 3D image array that is the result of divisively
    normalizing the 3D orientation-filtered image array values of the map data, whose keys should
    be the orientations (in radians) of the gabor filters applied to the values.

    Reference:
      Heeger DJ (1992) Vis. Neurosci. 9(2):181–197. doi:10.1017/S0952523800009640.
    '''
    surround = _np.mean(data.values(), axis=0)
    s = saturation_constant
    r = response_exponent
    den = (s**r + surround**r)
    num = _np.zeros(surround.shape)
    for v in data.itervalues(): num += v**r
    num /= len(data)
    normalized = num / div
    normalized.setflags(write=False)
    return normalized

# Parameters Defined by Labels #####################################################################
pRF_sigma_slope_by_label_Kay2013      = pyr.pmap({1:0.10, 2:0.15, 3:0.27})
contrast_constant_by_label_Kay2013    = pyr.pmap({1:0.93, 2:0.99, 3:0.99})
compressive_constant_by_label_Kay2013 = pyr.pmap({1:0.18, 2:0.13, 3:0.12})
saturation_constant_by_label_Kay2013  = pyr.pmap({1:0.50, 2:0.50, 3:0.50})
divisive_exponent_by_label_Kay2013    = pyr.pmap({1:1.00, 2:1.00, 3:1.00})

# Frequency Sensitivity ############################################################################
_cpd_sensitivity_frequencies = [0.75 * 2.0**(0.5 * k) for k in range(6)]
_cpd_sensitivity_std = 0.5
def cpd_sensitivity(e, s, l):
    '''
    cpd_sensitivity(ecc, prfsz, lbl) yields the predicted spatial frequency sensitivity of a 
    pRF whose center has the given eccentricity ecc, pRF radius prfsz, and V1/V2/V3 label lbl.
    This is returned as a map whose keys are sampled frequencies (in cycles per degree) and
    whose values sum to 1.
    '''
    # round to nearest 25th of a degree:
    s = round(25.0 * s) * 0.04
    if s in cpd_sensitivity._cache: return cpd_sensitivity._cache[s]
    ss = 1.5 / s
    weights = _np.asarray([_np.exp(-0.5*((k - ss)/_cpd_sensitivity_std)**2)
                          for k in _cpd_sensitivity_frequencies])
    wtot = _np.sum(weights)
    weights /= wtot
    res = {k:v for (k,v) in zip(_default_frequencies, weights) if v > 0.01}
    if len(res) < len(weights):
        wtot = _np.sum(res.values())
        res = {k:(v/wtot) for (k,v) in res.iteritems()}
    res = _pyr.pmap(res)
    cpd_sensitivity._cache[s] = res
    return res
cpd_sensitivity._cache = {}

# Default Options ##################################################################################
# The default options are provided here for the SCO
@pimms.calc('benson17_default_options_used')
def provide_default_options(
        pRF_sigma_slope_by_label      = pRF_sigma_slope_by_label_Kay2013,
        contrast_constant_by_label    = contrast_constant_by_label_Kay2013,
        compressive_constant_by_label = compressive_constant_by_label_Kay2013,
        saturation_constant           = saturation_constant_Kay2013,
        divisive_exponent             = divisive_exponent_Kay2013,
        max_eccentricity              = 12,
        modality                      = 'volume',
        cpd_sensitivity_function      = cpd_sensitivity,
        gabor_orientations            = 8):
    '''
    provide_default_options is a calculator that optionally accepts values for all parameters for
    which default values are provided in the sco.impl.benson17 package and yields into the calc plan
    these parameter values or the default ones for any not provided.
 
    These options are:
      * pRF_sigma_slope_by_label (sco.impl.benson17.pRF_sigma_slope_by_label_Kay2013)
      * compressive_constant_by_label (sco.impl.benson17.compressive_constant_by_label_Kay2013)
      * contrast_constant_by_label (sco.impl.benson17.contrast_constant_by_label_Kay2013)
      * modality ('volume')
      * max_eccentricity (12)
      * cpd_sensitivity_function (sco.impl.benson17.cpd_sensitivity)
      * saturation_constant (sco.impl.benson17.saturation_constant_Kay2013)
      * divisive_exponent (sco.impl.benson17.divisive_exponent_Kay2013)
      * gabor_orientations (8)
    '''
    # the defaults are filled-in by virtue of being in the above argument list
    return True
