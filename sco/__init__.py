# __init__.py

'''
The SCO Python library is a modular toolkit for predicting the response of the cortical surface to
visual stimuli.
'''

# Import relevant functions...
import pimms        as _pimms
import pyrsistent   as _pyr

from sco.util          import (cortical_image)
from sco.impl.benson17 import sco_plan as benson17_plan

sco_plans = {'benson17': benson17_plan}

#from sco.model_comparison import create_model_dataframe

# Version information...
_version_major = 0
_version_minor = 3
_version_micro = 0
__version__ = "%s.%s.%s" % (_version_major, _version_minor, _version_micro)

description = 'Predict the response of the cortex to visual stimuli'
    
# The volume (default) calculation chain
def reload_sco():
    '''
    reload_sco() reloads the sco and all of its submodules; it returns the new sco module.
    '''
    import sys
    reload(sys.modules['sco.util'])
    reload(sys.modules['sco.impl.benson17'])
    reload(sys.modules['sco.impl'])    
    reload(sys.modules['sco.anatomy.core'])
    reload(sys.modules['sco.stimulus.core'])
    reload(sys.modules['sco.contrast.core'])
    reload(sys.modules['sco.pRF.core'])
    reload(sys.modules['sco.anatomy'])
    reload(sys.modules['sco.stimulus'])
    reload(sys.modules['sco.pRF'])
    reload(sys.modules['sco.contrast'])
    reload(sys.modules['sco.analysis.core'])
    reload(sys.modules['sco.analysis'])
    return reload(sys.modules['sco'])

