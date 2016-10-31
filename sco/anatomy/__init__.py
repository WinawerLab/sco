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

from ..core import calc_chain
from .core  import (import_benson14_volumes_from_freesurfer,
                    import_benson14_surface_from_freesurfer,
                    calc_pRFs_from_freesurfer_retinotopy_volumes,
                    calc_pRFs_from_freesurfer_retinotopy_surface,
                    calc_anatomy_default_parameters,
                    calc_Kay2013_pRF_sizes,
                    export_predicted_response_volumes,
                    export_predicted_response_surface)

# Make a function that's ready to be used as a module
volumes_anatomy_chain = (
    ('import',                          import_benson14_volumes_from_freesurfer),
    ('calc_pRF_centers',                calc_pRFs_from_freesurfer_retinotopy_volumes),
    ('calc_anatomy_defualt_parameters', calc_anatomy_default_parameters),
    ('calc_pRF_sizes',                  calc_Kay2013_pRF_sizes))
surface_anatomy_chain = (
    ('import',                          import_benson14_surface_from_freesurfer),
    ('calc_pRF_centers',                calc_pRFs_from_freesurfer_retinotopy_surface),
    ('calc_anatomy_defualt_parameters', calc_anatomy_default_parameters),
    ('calc_pRF_sizes',                  calc_Kay2013_pRF_sizes))

calc_anatomy         = calc_chain(volumes_anatomy_chain)
calc_surface_anatomy = calc_chain(surface_anatomy_chain)

