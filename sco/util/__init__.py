####################################################################################################
# sco/util/__init__.py
# Utilities useful in work related to the sco predictions.
# By Noah C. Benson

import pyrsistent as _pyr
import pimms      as _pimms

from .plot import (cortical_image, corrcoef_image, report_image)
from .io   import (export_predictions, export_analysis, export_report_images, calc_exported_files,
                   require_exports)

# The unit registry that we use
units = _pimms.units

def lookup_labels(labels, data_by_labels, **kwargs):
    '''
    sco.util.lookup_labels(labels, data_by_labels) yields a list the same size as labels in which
      each element i of the list is equal to data_by_labels[labels[i]].
    
    The option null may additionally be passed to lookup_labels; if null is given, then whenever a
    label value from data is not found in labels, it is instead given the value null; if null is not
    given, then an error is raised in this situation.

    The lookup_labels function expects the labels to be integer or numerical values.
    '''
    res = None
    null = None
    raise_q = True
    if 'null' in kwargs:
        null = kwargs['null']
        raise_q = False
    if len(kwargs) > 1 or (len(kwargs) > 0 and 'null' not in kwargs):
        raise ValueError('Unexpected option given to lookup_labels; only null is accepted')
    if raise_q:
        try:
            res = [data_by_labels[lbl] for lbl in labels]
        except:
            raise ValueError('Not all labels found by lookup_labels and no null given')
    else:
        res = [data_by_labels[lbl] if lbl in data_by_labels else null for lbl in labels]
    return _pyr.pvector(res)

export_plan_data = _pyr.m(export_predictions     = export_predictions,
                          export_analysis        = export_analysis,
                          export_report_images   = export_report_images,
                          exported_files         = calc_exported_files)
export_plan = _pimms.plan(export_plan_data)


