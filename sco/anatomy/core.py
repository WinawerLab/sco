####################################################################################################
# anatomy/core.py
# The anatomy module of the standard cortical observer; core definitions and checks.
# By Noah C. Benson


import numpy                 as     np
import neuropythy            as     neuro
import neuropythy.freesurfer as     neurofs
import nibabel.freesurfer    as     fs

from   neuropythy.commands   import benson14_retinotopy_command

import os

from ..core            import calculates
from ..normalization   import _turn_param_into_array

@calculates()
def calc_anatomy_default_parameters(pRF_v123_labels,
                                    Kay2013_pRF_sigma_slope={1: 0.1,  2: 0.15, 3: 0.27},
                                    Kay2013_output_nonlinearity={1: 0.18, 2: 0.13, 3: 0.12}):
    """
    calc_anatomy_default_parameters() is a calculator that expects no particular options, but
    fills in several options if not provided:
    * Kay2013_pRF_sigma_slope
    * Kay2013_output_nonlinearity
      
    For both of these, they can be a single float, a list/array of float, or a dictionary with 1, 2
    and 3 as its keys and with floats as the values, specifying the values for these parameters for
    voxels in areas V1, V2, and V3. This function will take these values and form arrays that
    correspond to the other voxel-level arrays.

    Different from most of the calc_default_parameters functions, this one must run after
    calc_pRF_centers because it needs pRF_v123_labels
    """
    return {'Kay2013_pRF_sigma_slope': _turn_param_into_array(Kay2013_pRF_sigma_slope, pRF_v123_labels),
            'Kay2013_output_nonlinearity': _turn_param_into_array(Kay2013_output_nonlinearity, pRF_v123_labels)}

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
                                         max_eccentricity):
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
def calc_Kay2013_pRF_sizes(pRF_eccentricity, Kay2013_pRF_sigma_slope, Kay2013_output_nonlinearity):
    '''
    calculate_pRF_sizes_Kay2013(pRF_eccentricity, Kay2013_pRF_sigma_slope, Kay2013_output_nonlinearity) 
    yields a list of pRF sizes for the given lists of eccentricity values and labels (which should
    all be 1, 2, or 3 for V1-V3, otherwise 0). sigma0(p, a) = s0 + sa (p/90) where sa is a constant
    determined by the visual area and s0 is 1/2; the values for sa given in Kay et al. (2013) are
    {s1 = 0.1, s2 = 0.15, s3 = 0.27}.  sigma(p, a, n) = (sigma0(p, a) - 0.23) / (0.16 n^(1/2) +
    -0.05)
    '''
    sig0 = [(0.5 + sig_slope * e) for (e, sig_slope) in
            zip(pRF_eccentricity, Kay2013_pRF_sigma_slope)]
    sig = [(s0 - 0.23) / (0.16 / np.sqrt(nonlin) - 0.05) for (e, nonlin, s0) in
           zip(pRF_eccentricity, Kay2013_output_nonlinearity, sig0)]
    return {'pRF_sizes': np.asarray(sig)}


@calculates('exported_predictions_filename', 'exported_image_ordering_filename')
def export_predicted_responses(export_path,
                               predicted_responses,
                               pRF_voxel_indices, subject,
                               predicted_response_name='prediction',
                               image_order_name='images',
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
    # new method: one volume plus an image ordering text file
    preds = np.zeros(vol0dims + (len(predicted_responses),), dtype=np.float32)
    for (n,result) in enumerate(predicted_responses):
        for ((i,j,k),val) in zip(pRF_voxel_indices, result):
            preds[i,j,k,n] = val
    hdr = vol0.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr.set_data_shape(preds.shape)
    mgh = fs.mghformat.MGHImage(preds, vol0.affine, hdr, vol0.extra, vol0.file_map)
    mgh_flnm = os.path.join(export_path, predicted_response_name + '.mgz')
    mgh.to_filename(mgh_flnm)
    ord_flnm = os.path.join(export_path, image_order_name + '.txt')
    with open(ord_flnm, 'w') as f:
        for im in stimulus_file_names:
            f.write('%s\n' % im)
    return {'exported_predictions_filename':    mgh_flnm,
            'exported_image_ordering_filename': ord_flnm}
