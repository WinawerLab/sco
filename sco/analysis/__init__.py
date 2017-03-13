####################################################################################################
# sco/analysis/__init__.py
# Definition code for the analysis module of the sco library
# by Noah C. Benson

import pimms as _pimms
import pyrsistent as _pyr
from .core import (import_ground_truth, calc_prediction_analysis)

analysis_plan_data = _pyr.m(import_ground_truth = import_ground_truth,
                           prediction_analysis = calc_prediction_analysis)

analysis_plan = _pimms.plan(analysis_plan_data)

