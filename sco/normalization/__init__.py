##########################################################################
# sco/normalization/__init__.py
# Second-order-contrast normalization and nonlinearity application
# By Noah C. Benson

from .core  import (calc_Kay2013_SOC_normalization, calc_Kay2013_output_nonlinearity,
                    _turn_param_into_array, calc_normalization_default_parameters)
from ..core import calc_chain

normalization_chain = (('calc_normalization_default_parameters', calc_normalization_default_parameters),
                       ('calc_SOC_normalization',   calc_Kay2013_SOC_normalization),
                       ('calc_output_nonlinearity', calc_Kay2013_output_nonlinearity))

calc_normalization = calc_chain(normalization_chain)
