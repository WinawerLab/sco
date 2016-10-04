####################################################################################################
# anatomy/core.py
# The anatomy module of the standard cortical observer; core definitions and checks.
# By Noah C. Benson


import numpy                 as     np
import scipy                 as     sp
import neuropythy            as     neuro
import neuropythy.freesurfer as     neurofs
import nibabel.freesurfer    as     fs

from   neuropythy.immutable  import Immutable
from   neuropythy.commands   import benson14_retinotopy_command
from   numbers               import Number
from   pysistence            import make_dict

import os, math, itertools, collections, abc

from ..core            import (calculates, calc_chain, iscalc)

@calculates('polar_angle_mgh', 'eccentricity_mgh', 'v123_labels_mgh', 'ribbon_mghs')
def import_benson14_from_freesurfer(subject):
    if isinstance(subject, basestring):
        subject = neuro.freesurfer_subject(subject)
    if not isinstance(subject, neurofs.Subject):
        raise ValueError('Subject given to FreeSurferAnatomyModule object is not a neuropythy' + \
                         ' FreeSurfer subject or string')
    # make sure there are template volume files that match this subject
    ang = os.path.join(subject.directory, 'mri', 'angle_benson14.mgz')
    ecc = os.path.join(subject.directory, 'mri', 'eccen_benson14.mgz')
    lab = os.path.join(subject.directory, 'mri', 'v123roi_benson14.mgz')
    if not os.path.exists(ang) or not os.path.exists(ecc) or not os.path.exists(lab):
        # Apply the template first...
        benson14_retinotopy_command(subject.directory)
    if not os.path.exists(ang) or not os.path.exists(ecc) or not os.path.exists(lab):        
        raise ValueError('No areas template found/created for subject: ' + lab)
    angle_mgz = fs.mghformat.load(ang)
    eccen_mgz = fs.mghformat.load(ecc)
    label_mgz = fs.mghformat.load(lab)
    ribbon_mgzs = (subject.LH.ribbon, subject.RH.ribbon)
    return {'subject':          subject,
            'polar_angle_mgh':  angle_mgz,
            'eccentricity_mgh': eccen_mgz,
            'v123_labels_mgh':  label_mgz,
            'ribbon_mghs':       ribbon_mgzs}

@calculates('pRF_centers', 'pRF_voxel_indices',
            'pRF_polar_angle', 'pRF_eccentricity', 'pRF_v123_labels')
def calc_pRFs_from_freesurfer_retinotopy(polar_angle_mgh, eccentricity_mgh,
                                         v123_labels_mgh, ribbon_mghs,
                                         max_eccentricity=None):
    # The variables are all mgz volumes, so we need to extract the values:
    label = np.round(np.abs(v123_labels_mgh.dataobj.get_unscaled()))
    angle = polar_angle_mgh.dataobj.get_unscaled()
    eccen = eccentricity_mgh.dataobj.get_unscaled()
    (lrib, rrib) = [r.dataobj.get_unscaled() for r in ribbon_mghs]
    # Find the voxel indices first:
    pRF_voxel_indices = np.asarray(np.where(label.astype(bool))).T
    # Grab the hemispheres; filter down if something isn't in the ribbon
    tmp   = [(1 if lrib[i,j,k] == 1 else -1, (i,j,k))
             for (i,j,k) in pRF_voxel_indices
             if lrib[i,j,k] != 0 or rrib[i,j,k] != 0]
    [hem, pRF_voxel_indices] = [np.asarray([r[i] for r in tmp]) for i in [0,1]]
    # Pull out the angle/eccen data
    angs0 = np.asarray([angle[i,j,k] for (i,j,k) in pRF_voxel_indices])
    angs  = 180.0/np.pi * (90.0 - angs0*hem)
    eccs  = np.asarray([eccen[i,j,k] for (i,j,k) in pRF_voxel_indices])
    labs  = np.asarray([label[i,j,k] for (i,j,k) in pRF_voxel_indices])
    # Filter by eccentricity...
    subidcs = np.where(eccs < max_eccentricity)[0]
    angs = angs[subidcs]
    eccs = eccs[subidcs]
    labs = labs[subidcs]
    pRF_voxel_indices = pRF_voxel_indices[subidcs]
    hem = hem[subidcs]
    # and pull out the rest of the data based on these:
    return {'pRF_centers':       np.asarray([eccs * np.cos(angs), eccs * np.sin(angs)]).T,
            'pRF_voxel_indices': pRF_voxel_indices,
            'pRF_polar_angle':   angs,
            'pRF_eccentricity':  eccs,
            'pRF_v123_labels':   labs,
            'pRF_hemispheres':   hem,
            'max_eccentricity':  max_eccentricity}

@calculates('pRF_sizes')
def calc_Kay2013_pRF_sizes(pRF_eccentricity, pRF_v123_labels,
                           Kay2013_pRF_sigma_slope={1: 0.1,  2: 0.15, 3: 0.27},
                           Kay2013_output_nonlinearity={1: 0.18, 2: 0.13, 3: 0.12}):
    '''
    calculate_pRF_sizes_Kay2013(pRF_eccentricity, pRF_v123_labels) yeilds a list of pRF sizes for
    the given lists of eccentricity values and labels (which should all be 1, 2, or 3 for V1-V3,
    otherwise 0). For any label that isn't 1, 2, or 3, this function returns 0.
    sigma0(p, a) = s0 + sa (p/90) where sa is a constant determined by the visual area and s0 is
    1/2; the values for sa given in Kay et al. (2013) are {s1 = 0.1, s2 = 0.15, s3 = 0.27}.
    sigma(p, a, n) = (sigma0(p, a) - 0.23) / (0.16 n^(1/2) + -0.05)
    '''
    sig0 = [(0.5 + Kay2013_pRF_sigma_slope[l] * e) if l in Kay2013_pRF_sigma_slope else 0
            for (e, l) in zip(pRF_eccentricity, pRF_v123_labels)]
    sig  = [0 if l not in Kay2013_output_nonlinearity else \
            (s0 - 0.23) / (0.16 / np.sqrt(Kay2013_output_nonlinearity[l]) - 0.05)
            for (e, l, s0) in zip(pRF_eccentricity, pRF_v123_labels, sig0)]
    return {'pRF_sizes': sig}


@calculates()
def export_predicted_responses(export_path,
                               predicted_responses,
                               pRF_voxel_indices, subject,
                               predicted_response_names='prediction_',
                               exported_predictions={}, overwrite_files=True,
                               voxel_fill_value=0):
    '''
    export_predictions is a calculation module that expects the following data:
     * export_path (where to put the files)
     * predictions (dictionary whose keys are names and whose values are predictions)
     * pRF_voxel_indices (voxel indices at which the predictions are relevant)
     * subject (the subject whose data is being exported)
    Additionally, the following may be provided optionally:
     * exported_predictions (a dictionary of previously exported filed, default {})
     * overwrite_files (whether to overwrite existing files, default True)
    The module provides one additional output, exported_predictions, which is a dictionary whose
    keys are the prediction names and whose values are the filenames of those predictions that have
    been successfully exported.
    '''
    fill = voxel_fill_value
    vol0 = subject.LH.ribbon
    vol0dims = vol0.get_data().shape
    if isinstance(predicted_response_names, basestring):
        predicted_response_names = ['%s%04d' % (predicted_response_names, i)
                                    for i in range(len(predicted_responses))]
    for (nm, result) in zip(predicted_response_names, predicted_responses):
        flname = os.path.join(export_path, nm + '.mgz')
        if os.path.isdir(flname): raise ValueError('Output filename %s is a directory!' % flname)
        if os.path.exists(flname) and not overwrite_files: continue
        if len(result) != len(pRF_voxel_indices):
            raise ValueError('data (%s) is not the same length as pRF_voxel_indices!' % nm)
        arr = fill * np.ones(vol0dims) if fill != 0 else np.zeros(vol0dims)
        for ((i,j,k),val) in zip(pRF_voxel_indices, result):
            arr[i,j,k] = val
        hdr = vol0.header.copy()
        hdr.set_data_dtype(np.float32)
        mgh = fs.mghformat.MGHImage(arr, vol0.affine, hdr, vol0.extra, vol0.file_map)
        mgh.to_filename(flname)
        exported_predictions[nm] = flname
    return {'exported_predictions': exported_predictions}
