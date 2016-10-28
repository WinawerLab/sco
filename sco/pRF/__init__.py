####################################################################################################
# sco/pRF/__init__.py
# pRF-related calculations for the standard cortical observer library.
# By Noah C. Benson

from ..core import (calc_chain)
from .core  import (calc_pRF_default_parameters, calc_pRF_views, PRFSpec,
                    calc_pRF_frequency_preferences)

pRF_chain = (('calc_pRF_default_parameters',    calc_pRF_default_parameters),
             ('calc_pRF_frequency_preferences', calc_pRF_frequency_preferences),
             ('calc_pRF_views',                 calc_pRF_views))

calc_pRF = calc_chain(pRF_chain)

