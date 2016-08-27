# __init__.py

'''
The SCO Python library is a modular toolkit for predicting the response of the cortical surface to
visual stimuli.
'''

# Import relevant functions...
from .core import (iscalc, calculates, calc_chain)

# Version information...
_version_major = 0
_version_minor = 1
_version_micro = 0
__version__ = "%s.%s.%s" % (_version_major, _version_minor, _version_micro)

description = 'Predict the response of the cortex to visual stimuli'
    
