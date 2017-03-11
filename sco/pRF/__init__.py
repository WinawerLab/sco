####################################################################################################
# sco/pRF/__init__.py
# pRF-related calculations for the standard cortical observer library.
# By Noah C. Benson

'''
The sco.pRF module contains calculation plans for producing PRFSpec objects, which track the data
for each pRF involved in the sco calculation and can extract and sum over regions from images.
'''

import pyrsistent as pyr
import pimms
from .core  import (calc_pRFs, calc_cpd_sensitivities, PRFSpec)

pRF_plan_data = pyr.m(calc_pRFs             = calc_pRFs,
                      calc_cpd_sensitivites = calc_cpd_sensitivities)

pRF_plan = pimms.plan(pRF_plan_data)

