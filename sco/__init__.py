# __init__.py

'''
The SCO Python library is a modular toolkit for predicting the response of the cortical surface to
visual stimuli.
'''

# Import relevant functions...
import pimms as _pimms
import pyrsistent as _pyr

import sco.stimulus as stimulus
import sco.contrast as contrast
import sco.pRF      as pRF
import sco.anatomy  as anatomy

from sco.util import (cortical_image)

#from sco.model_comparison import create_model_dataframe

# Version information...
_version_major = 0
_version_minor = 3
_version_micro = 0
__version__ = "%s.%s.%s" % (_version_major, _version_minor, _version_micro)

description = 'Predict the response of the cortex to visual stimuli'
    
# The volume (default) calculation chain
sco_plan_data = _pyr.m(anatomy  = anatomy.anatomy_plan,
                       stimulus = stimulus.stimulus_plan,
                       contrast = contrast.contrast_plan,
                       pRF      = pRF.pRF_plan)

sco_plan         = _pimms.plan(sco_chain)

def reload_sco():
    '''
    reload_sco() reloads the sco and all of its submodules; it returns the new sco module.
    '''
    import sys
    reload(sys.modules['sco.anatomy.core'])
    reload(sys.modules['sco.stimulus.core'])
    reload(sys.modules['sco.contrast.core'])
    reload(sys.modules['sco.pRF.core'])
    reload(sys.modules['sco.anatomy'])
    reload(sys.modules['sco.stimulus'])
    reload(sys.modules['sco.contrast'])
    reload(sys.modules['sco.pRF'])
    reload(sys.modules['sco.util'])
    return reload(sys.modules['sco'])

