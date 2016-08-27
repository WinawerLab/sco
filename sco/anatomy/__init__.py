####################################################################################################
# anatomy/__init__.py
# The anatomy module of the standard cortical observer.
# By Noah C. Benson

'''
The sco.anatomy module of the standard cortical observer library is responsible for importing and
interpreting anatomical data from a subject and providing (and tracking) a list of parameters for
each voxel. Additionally, the anatomy module includes utilities for writing out results as a
FreeSurfer volume.

The anatomy module defines an abstract base class, AnatomyBase, whose abstract methods define the
interface for the module. When constructing an SCO pipeline, an AnatomyBase object is required,
and one may be obtained from the StandardAnatomy class or from a custom class that overloads the
AnatomyBase class.
'''

import sco
from .core import (import_freesurfer_subject_benson14, freesurfer_retinotopy_to_pRFs,
                   calculate_pRF_sizes_Kay2013,
                   Kay2013_pRF_sigma_slope, Kay2013_pRF_output_nonlinearities)

# Make a function that's ready to be used as a module
import_freesurfer_pRF_chain = (import_freesurfer_subject_benson14,
                               freesurfer_retinotopy_to_pRFs,
                               calculate_pRF_sizes_Kay2013)
import_freesurfer_pRFs = sco.calc_chain(import_freesurfer_pRF_chain)
