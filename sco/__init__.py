# __init__.py

'''
The SCO Python library is a modular toolkit for predicting the response of the cortical surface to
visual stimuli.
'''

# Import relevant functions...
import pimms        as _pimms
import pyrsistent   as _pyr

from sco.stimulus import (stimulus_plan, stimulus_plan_data)
from sco.contrast import (contrast_plan, contrast_plan_data)
from sco.pRF      import (pRF_plan,      pRF_plan_data)
from sco.anatomy  import (anatomy_plan,  anatomy_plan_data)

from sco.util import (cortical_image)

import sco.impl.benson17

#from sco.model_comparison import create_model_dataframe

# Version information...
_version_major = 0
_version_minor = 3
_version_micro = 0
__version__ = "%s.%s.%s" % (_version_major, _version_minor, _version_micro)

description = 'Predict the response of the cortex to visual stimuli'
    
# The volume (default) calculation chain
sco_plan_data = _pyr.pmap({k:v
                           for pd    in [stimulus_plan_data, contrast_plan_data,
                                         pRF_plan_data,      anatomy_plan_data,
                                         {'provide_default_options':
                                          sco.impl.benson17.provide_default_options}]
                           for (k,v) in pd.iteritems()})

sco_plan         = _pimms.plan(sco_plan_data)

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
    reload(sys.modules['sco.contrast'])
    reload(sys.modules['sco.pRF'])
    return reload(sys.modules['sco'])

