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

import pyrsistent              as _pyr
import pimms                   as _pimms
import numpy                   as _np
from   sco.util   import units as _units

import sco.anatomy
import sco.stimulus
import sco.pRF
import sco.contrast
import sco.analysis
import sco.util


# Divisive Normalization ###########################################################################
# The following are divisive normalization models that can be used with the SCO
def divisively_normalize_Heeger1992(data, divisive_exponent=0.5, saturation_constant=1.0):
    '''
    divisively_normalize(data) yields the 3D image array that is the result of divisively
    normalizing the 3D orientation-filtered image array values of the map data, whose keys should
    be the orientations (in radians) of the gabor filters applied to the values.

    Reference:
      Heeger DJ (1992) Vis. Neurosci. 9(2):181-197. doi:10.1017/S0952523800009640.
    '''
    surround = _np.mean(data.values(), axis=0)
    s = saturation_constant
    r = divisive_exponent
    den = (s**r + surround**r)
    num = _np.zeros(surround.shape)
    for v in data.itervalues(): num += v**r
    num /= len(data)
    normalized = num / den
    normalized.setflags(write=False)
    return normalized

@_pimms.calc('divisive_normalization_parameters', 'divisive_normalization_function')
def calc_divisive_normalization(labels, saturation_constants_by_label, divisive_exponents_by_label):
    '''
    calc_divisive_normalization is a calculator that prepares the divisive normalization function
    to be run in the sco pipeline. It gathers parameters into a pimms itable (such that each row
    is a map of the parameters for each pRF in the pRFs list), which is returned as the value
    'divisive_normalization_parameters'; it also adds a 'divisive_normalization_function' that
    is appropriate for the parameters given. In the case of this implementation, the parameters
    saturation_constant and divisive_exponent are extracted from the afferent parameters
    saturation_constant_by_label and divisive_exponent_by_label, and the function
    sco.impl.benson17.divisively_normalize_Heeger1992 is used.

    Required afferent parameters:
      * labels
      @ saturation_constants_by_label Must be a map whose keys are label values and whose values are
        the saturation constant for the particular area; all values appearing in the pRF labels
        must be found in this map.
      * divisive_exponents_by_label Must be a map whose keys are label values and whose values are
        the divisive normalization exponent for that particular area; all values appearing in the
        pRF labels must be found in this map.

    Provided efferent values:
      @ divisive_normalization_parameters Will be an ITable whose columns correspond to the
        divisive normalization formula's saturation constant and exponent; the rows will correspond
        to the pRFs.
      @ divisive_normalization_function Will be a function compatible with the
        divisive_normalization_parameters data-table; currently this is
        sco.impl.benson17.divisively_normalize_Heeger1992.
    '''
    sat = sco.util.lookup_labels(labels, saturation_constants_by_label)
    rxp = sco.util.lookup_labels(labels, divisive_exponents_by_label)
    return (_pimms.itable(saturation_constant=sat, divisive_exponent=rxp),
            divisively_normalize_Heeger1992)

# Parameters Defined by Labels #####################################################################
pRF_sigma_slopes_by_label_Kay2013      = _pyr.pmap({1:0.10, 2:0.15, 3:0.27})
contrast_constants_by_label_Kay2013    = _pyr.pmap({1:0.93, 2:0.99, 3:0.99})
compressive_constants_by_label_Kay2013 = _pyr.pmap({1:0.18, 2:0.13, 3:0.12})
saturation_constants_by_label_Kay2013  = _pyr.pmap({1:0.50, 2:0.50, 3:0.50})
divisive_exponents_by_label_Kay2013    = _pyr.pmap({1:1.00, 2:1.00, 3:1.00})

# Frequency Sensitivity ############################################################################
_sensitivity_frequencies_cpd = _pimms.quant(_np.asarray([0.75 * 2.0**(0.5 * k) for k in range(6)]),
                                            'cycles/deg')
_sensitivity_std_cpd         = 0.5
def cpd_sensitivity(e, s, l):
    '''
    cpd_sensitivity(ecc, prfsz, lbl) yields the predicted spatial frequency sensitivity of a 
    pRF whose center has the given eccentricity ecc, pRF radius prfsz, and V1/V2/V3 label lbl.
    This is returned as a map whose keys are sampled frequencies (in cycles per degree) and
    whose values sum to 1.
    '''
    s = _pimms.mag(s, 'deg')
    if s in cpd_sensitivity._cache: return cpd_sensitivity._cache[s]
    ss = 1.5 / s
    weights = _np.asarray([_np.exp(-0.5*((k.m - ss)/_sensitivity_std_cpd)**2)
                           for k in _sensitivity_frequencies_cpd])
    wtot = _np.sum(weights)
    weights /= wtot
    res = {k:v for (k,v) in zip(_sensitivity_frequencies_cpd, weights) if v > 0.01}
    if len(res) < len(weights):
        wtot = _np.sum(res.values())
        res = {k:(v/wtot) for (k,v) in res.iteritems()}
    res = _pyr.pmap(res)
    cpd_sensitivity._cache[s] = res
    return res
cpd_sensitivity._cache = {}

# Default Options ##################################################################################
# The default options are provided here for the SCO
@_pimms.calc('benson17_default_options_used')
def provide_default_options(
        pRF_sigma_slopes_by_label      = pRF_sigma_slopes_by_label_Kay2013,
        contrast_constants_by_label    = contrast_constants_by_label_Kay2013,
        compressive_constants_by_label = compressive_constants_by_label_Kay2013,
        saturation_constants_by_label  = saturation_constants_by_label_Kay2013,
        divisive_exponents_by_label    = divisive_exponents_by_label_Kay2013,
        max_eccentricity               = 12,
        modality                       = 'volume',
        cpd_sensitivity_function       = cpd_sensitivity,
        gabor_orientations             = 8):
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

# The volume (default) calculation chain
sco_plan_data = _pyr.pmap({k:v
                           for pd    in [sco.stimulus.stimulus_plan_data,
                                         sco.contrast.contrast_plan_data,
                                         sco.pRF.pRF_plan_data,
                                         sco.anatomy.anatomy_plan_data,
                                         sco.analysis.analysis_plan_data,
                                         {'default_options': provide_default_options,
                                          'divisive_normalization': calc_divisive_normalization}]
                           for (k,v) in pd.iteritems()})

sco_plan      = _pimms.plan(sco_plan_data)
