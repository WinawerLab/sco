####################################################################################################
# impl/options/__init__.py
# This file contains default values for the particular implementation of the sco model that is given
# by the sco.impl namespace.
# By Noah C. Benson

'''
The sco.impl.options namespace contains default values for all of the optional parameters used in
the Winawer lab implementation of the SCO model. A function, provide_default_options is also
defined; this may be included in any SCO plan to ensure that the optional parameter values are
available to the downstream calculations if not provided by the user or other calculations.
'''

import pimms

pRF_sigma_slope_by_label_Kay2013     = pyr.pmap({1:0.1,  2:0.15, 3:0.27})
output_nonlinearity_by_label_Kay2013 = pyr.pmap({1:0.18, 2:0.13, 3:0.12})

@pimms.calc('pRF_sigma_slope_by_label', 'output_nonlinearity_by_label')
def provide_winawerlab_default_options(
        pRF_sigma_slope_by_label=pRF_sigma_slope_by_label_Kay2013,
        output_nonlinearity_by_label=output_nonlinearity_by_label_Kay2013):
    '''
    provide_winawerlab_default_options is a calculator that optionally accepts values for all 
    parameters for which default values are provided in the sco.impl.options package and yields into
    the calc plan these parameter values or the default ones for any not provided.
 
    These options are:
      * pRF_sigma_slope_by_label (sco.impl.options.pRF_sigma_slope_by_label_Kay2013)
      * output_nonlinearity_by_label (sco.impl.options.output_nonlinearity_by_label_Kay2013)
    '''
    return {'pRF_sigma_slope_by_label': pRF_sigma_slope_by_label,
            'output_nonlinearity_by_label': output_nonlinearity_by_label}

