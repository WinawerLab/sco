# __init__.py

'''
The SCO Python library is a modular toolkit for predicting the response of the cortical surface to
visual stimuli.
'''

# Import relevant functions...
from .core import (iscalc, calculates, calc_chain, calc_translate)

from sco.anatomy       import (calc_anatomy,       anatomy_chain, export_predicted_responses)
from sco.stimulus      import (calc_stimulus,      stimulus_chain)
from sco.pRF           import (calc_pRF,           pRF_chain)
from sco.normalization import (calc_normalization, normalization_chain)

# Version information...
_version_major = 0
_version_minor = 1
_version_micro = 0
__version__ = "%s.%s.%s" % (_version_major, _version_minor, _version_micro)

description = 'Predict the response of the cortex to visual stimuli'
    
# The default calculation chain
sco_chain = (('calc_anatomy',       calc_anatomy),
             ('calc_stimulus',      calc_stimulus),
             ('calc_pRF',           calc_pRF),
             ('calc_normalization', calc_normalization))

calc_sco = calc_chain(sco_chain)
