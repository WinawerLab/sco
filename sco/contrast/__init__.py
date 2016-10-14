####################################################################################################
# contrast/__init__.py
# The second-order contrast module of the standard cortical observer.
# By Noah C. Benson

'''
The sco.contrast module of the standard cortical observer library is responsible for calculating the
second order contrast present in the normalized stimulus images. In general, these functions work
in a lazy fashion, by returning a function that, when called, calculates and returns the result.
'''

from ..core import calc_chain
from .core  import (calc_contrast_default_parameters, calc_stimulus_contrast_functions,
                    calc_divisive_normalization_functions)

contrast_chain = (('calc_contrast_default_parameters', calc_contrast_default_parameters),
                  ('calc_stimulus_contrast_functions', calc_stimulus_contrast_functions),
                  ('calc_divisive_normalization_functions', calc_divisive_normalization_functions))

calc_contrast = calc_chain(contrast_chain)


