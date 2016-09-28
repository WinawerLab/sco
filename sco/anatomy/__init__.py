##########################################################################
# anatomy/__init__.py
# The anatomy module of the standard cortical observer.
# By Noah C. Benson

''' The sco.anatomy module of the standard cortical observer library
is responsible for importing and interpreting anatomical data from a
subject and providing (and tracking) a list of parameters for each
voxel. Additionally, the anatomy module includes utilities for writing
out results as a FreeSurfer volume.

The anatomy module defines an abstract base class, AnatomyBase, whose
abstract methods define the interface for the module. When
constructing an SCO pipeline, an AnatomyBase object is required, and
one may be obtained from the StandardAnatomy class or from a custom
class that overloads the AnatomyBase class.  '''

import sco
from ..core import calc_chain
from .core import (import_benson14_from_freesurfer,
                   calc_pRFs_from_freesurfer_retinotopy,
                   calc_Kay2013_pRF_sizes, export_predicted_responses)

# Make a function that's ready to be used as a module
anatomy_chain = (('import',           import_benson14_from_freesurfer),
                 ('calc_pRF_centers', calc_pRFs_from_freesurfer_retinotopy),
                 ('calc_pRF_sizes',   calc_Kay2013_pRF_sizes))
calc_anatomy = calc_chain(anatomy_chain)
