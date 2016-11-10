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
    return {'Kay2013_pRF_sigma_slope': _turn_param_into_array(Kay2013_pRF_sigma_slope,
                                                              pRF_v123_labels),
            'Kay2013_output_nonlinearity': _turn_param_into_array(Kay2013_output_nonlinearity,
                                                                  pRF_v123_labels)}

@calculates('polar_angle_mgh', 'eccentricity_mgh', 'v123_labels_mgh', 'ribbon_mghs')
def import_benson14_volumes_from_freesurfer(subject):
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
            'ribbon_mghs':      ribbon_mgzs}

@calculates('lh_polar_angle_mgh', 'lh_eccentricity_mgh', 'lh_v123_labels_mgh',
            'rh_polar_angle_mgh', 'rh_eccentricity_mgh', 'rh_v123_labels_mgh')
def import_benson14_surface_from_freesurfer(subject):
    if isinstance(subject, basestring):
        subject = neuro.freesurfer_subject(subject)
    if not isinstance(subject, neurofs.Subject):
        raise ValueError('Subject given to FreeSurferAnatomyModule object is not a neuropythy' + \
                         ' FreeSurfer subject or string')
    # make sure there are template volume files that match this subject
    lang = os.path.join(subject.directory, 'surf', 'lh.angle_benson14.mgz')
    lecc = os.path.join(subject.directory, 'surf', 'lh.eccen_benson14.mgz')
    llab = os.path.join(subject.directory, 'surf', 'lh.v123roi_benson14.mgz')
    rang = os.path.join(subject.directory, 'surf', 'rh.angle_benson14.mgz')
    recc = os.path.join(subject.directory, 'surf', 'rh.eccen_benson14.mgz')
    rlab = os.path.join(subject.directory, 'surf', 'rh.v123roi_benson14.mgz')
    if not os.path.exists(lang) or not os.path.exists(lecc) or not os.path.exists(llab) \
       or not os.path.exists(rang) or not os.path.exists(recc) or not os.path.exists(rlab):
        # Apply the template first...
        benson14_retinotopy_command(subject.directory)
    if not os.path.exists(lang) or not os.path.exists(lecc) or not os.path.exists(llab) \
       or not os.path.exists(rang) or not os.path.exists(recc) or not os.path.exists(rlab):
        raise ValueError('No areas template found/created for subject: ' + lab)
    l_angle_mgz = fs.mghformat.load(lang)
    l_eccen_mgz = fs.mghformat.load(lecc)
    l_label_mgz = fs.mghformat.load(llab)
    r_angle_mgz = fs.mghformat.load(rang)
    r_eccen_mgz = fs.mghformat.load(recc)
    r_label_mgz = fs.mghformat.load(rlab)
    return {'subject':             subject,
            'lh_polar_angle_mgh':  l_angle_mgz,
            'lh_eccentricity_mgh': l_eccen_mgz,
            'lh_v123_labels_mgh':  l_label_mgz,
            'rh_polar_angle_mgh':  r_angle_mgz,
            'rh_eccentricity_mgh': r_eccen_mgz,
            'rh_v123_labels_mgh':  r_label_mgz}

@calculates('pRF_centers', 'pRF_voxel_indices', 'pRF_hemispheres',
            'pRF_polar_angle', 'pRF_eccentricity', 'pRF_v123_labels')
def calc_pRFs_from_freesurfer_retinotopy_volumes(polar_angle_mgh, eccentricity_mgh,
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
            'pRF_hemispheres':   hem}

@calculates('pRF_centers', 'pRF_hemispheres', 'pRF_vertex_indices',
            'pRF_eccentricity', 'pRF_polar_angle', 'pRF_v123_labels')
def calc_pRFs_from_freesurfer_retinotopy_surface(lh_polar_angle_mgh,
                                                 lh_eccentricity_mgh,
                                                 lh_v123_labels_mgh,
                                                 rh_polar_angle_mgh,
                                                 rh_eccentricity_mgh,
                                                 rh_v123_labels_mgh,
                                                 max_eccentricity):
    # The variables are all already vectors so this is pretty simple:
    (lang, lecc, llab)  = [mgh.dataobj.get_unscaled().flatten()
                           for mgh in [lh_polar_angle_mgh, lh_eccentricity_mgh, lh_v123_labels_mgh]]
    (rang, recc, rlab)  = [mgh.dataobj.get_unscaled().flatten()
                           for mgh in [rh_polar_angle_mgh, rh_eccentricity_mgh, rh_v123_labels_mgh]]
    (angs0, eccs, labs) = [np.concatenate([ldat, rdat], axis=0)
                           for (ldat,rdat) in zip([lang, lecc, llab], [rang, recc, rlab])]
    idcs  = np.concatenate([range(len(lang)), range(len(rang))], axis=0)
    vals  = np.intersect1d(np.intersect1d(np.where(labs > 0)[0],
                                          np.where(labs < 4)[0]),
                           np.where(eccs < max_eccentricity)[0])
    idcs  = idcs[vals]
    hemis = np.concatenate([[1 for a in lang], [-1 for a in rang]], axis=0)[vals]
    angs  = 180.0/np.pi * (90.0 - angs0[vals]*hemis)
    eccs  = eccs[vals]
    labs  = labs[vals]
    # and pull out the rest of the data based on these:
    return {'pRF_centers':        np.asarray([eccs * np.cos(angs), eccs * np.sin(angs)]).T,
            'pRF_vertex_indices': idcs,
            'pRF_polar_angle':    angs,
            'pRF_eccentricity':   eccs,
            'pRF_v123_labels':    labs,
            'pRF_hemispheres':    hemis}

@calculates()
def calc_voxel_selector(voxel_idx, pRF_centers, pRF_voxel_indices, pRF_polar_angle,
                        pRF_eccentricity, pRF_v123_labels, pRF_hemispheres):
    """
    calc_voxel_selector takes in the various pRF arrays that are retrieved from the anatomical
    image and retinotopy and limits them, selecting only some of the voxels. This is an optional
    step, not included in the default chain, that allows the user to fit the model and predict
    responses for only a subset of the voxels in the brain. This is helpful for debugging when one
    wants to examine a small number of voxels and many images or when one knows exactly what voxel
    they're interested in. As such, it should be inserted happen after
    calc_pRFs_from_freesurfer_retinotopy and before calc_anatomy_default_parameters and
    calc_Kay2013_pRF_sizes

    As an argument, it takes voxel_idx, a list or array containing ints; we use it as an index into
    the voxel-related arrays.
    """
    voxel_idx = np.asarray(voxel_idx)
    return {'pRF_centers':       pRF_centers[voxel_idx],
            'pRF_voxel_indices': pRF_voxel_indices[voxel_idx],
            'pRF_polar_angle':   pRF_polar_angle[voxel_idx],
            'pRF_eccentricity':  pRF_eccentricity[voxel_idx],
            'pRF_v123_labels':   pRF_v123_labels[voxel_idx],
            'pRF_hemispheres':   pRF_hemispheres[voxel_idx]}

@calculates('pRF_sizes')
def calc_Kay2013_pRF_sizes(pRF_eccentricity, Kay2013_pRF_sigma_slope, Kay2013_output_nonlinearity,
                           pRF_v123_labels):
    '''
    calculate_pRF_sizes_Kay2013 is a calculator object that requires the pRF_eccentricity,
    Kay2013_pRF_sigma_slope, and Kay2013_output_nonlinearity parameters; it yields a list of pRF
    sizes for the given lists of eccentricity values and labels (which should all be 1, 2, or 3
    for V1-V3, otherwise 0). sigma0(p, a) = s0 + sa (p/90) where sa is a constant determined by the
    visual area and s0 is 1/2; the values for sa given in Kay et al. (2013) are 
    {s1 = 0.1, s2 = 0.15, s3 = 0.27}.  sigma(p, a, n) = (sigma0(p, a) - 0.23) / (0.16 n^(1/2) +
    -0.05)
    '''
    # #TODO Note that the above text may be wrong now; we have to rethink the pRF size.
    sig0 = [(0.5 + sig_slope*e) for (e,sig_slope) in zip(pRF_eccentricity, Kay2013_pRF_sigma_slope)]
    sig1 = sig0 / np.sqrt(Kay2013_output_nonlinearity)
    m = {1: 1.243, 2: 1.313, 3: 1.618}
    b = {1: 0.282, 2: 0.336, 3: -0.321}
    sig2 = np.asarray([(s - b[lab])/m[lab] for (s,lab) in zip(sig1, pRF_v123_labels)])
    #sig = [(s0 - 0.23) / (0.16 / np.sqrt(nonlin) - 0.05) for (e, nonlin, s0) in
    #       zip(pRF_eccentricity, Kay2013_output_nonlinearity, sig0)]
    return {'pRF_sizes': sig2}


@calculates('exported_predictions_filename', 'exported_image_ordering_filename')
def export_predicted_response_volumes(export_path,
                                      predicted_responses,
                                      pRF_voxel_indices, subject,
                                      predicted_response_name='prediction',
                                      image_order_name='images',
                                      stimulus_image_filenames=None,
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
        if stimulus_image_filenames is None:
            f.write('# No filenames were given to export_predictions; this usually indicates\n')
            f.write('# that the sco calculation was run on a dataset imported directly into\n')
            f.write('# Python instead of via filenames.\n')
        else:
            for im in stimulus_image_filenames:
                f.write('%s\n' % im)
    return {'exported_predictions_filename':    mgh_flnm,
            'exported_image_ordering_filename': ord_flnm}

@calculates('lh_exported_predictions_filename',
            'rh_exported_predictions_filename',
            'exported_image_ordering_filename')
def export_predicted_response_surface(export_path,
                                      predicted_responses,
                                      subject,
                                      pRF_vertex_indices,
                                      pRF_hemispheres,
                                      predicted_response_name='prediction',
                                      image_order_name='images',
                                      stimulus_image_filenames=None,
                                      exported_predictions={}, overwrite_files=True,
                                      vertex_fill_value=0):
    '''
    export_predictions is a calculation module that expects the following data:
     * export_path (where to put the files)
     * predictions (dictionary whose keys are names and whose values are predictions)
     * pRF_vertex_indices (voxel indices at which the predictions are relevant)
     * pRF_vertex_hemisphere
     * subject (the subject whose data is being exported)
    Additionally, the following may be provided optionally:
     * exported_predictions (a dictionary of previously exported filed, default {})
     * overwrite_files (whether to overwrite existing files, default True)
    The module provides one additional output, exported_predictions, which is a dictionary whose
    keys are the prediction names and whose values are the filenames of those predictions that have
    been successfully exported.
    '''
    fill = vertex_fill_value
    flnms = []
    for (hemi, hid) in zip([subject.LH, subject.RH], [1, -1]):
        preds = np.full((1, 1, hemi.vertex_count, predicted_responses.shape[0]), float(fill),
                        dtype=np.float32)
        idcs  = np.where(pRF_hemispheres == hid)[0]
        pidcs = pRF_vertex_indices[idcs]
        preds[0,0,pidcs,:] = predicted_responses[:,idcs].T
        mgh = fs.mghformat.MGHImage(preds, np.eye(4))
        mgh_flnm = os.path.join(export_path,
                                hemi.name.lower() + '.' + predicted_response_name + '.mgz')
        mgh.to_filename(mgh_flnm)
        flnms.append(mgh_flnm)
    ord_flnm = os.path.join(export_path, image_order_name + '.txt')
    with open(ord_flnm, 'w') as f:
        if stimulus_image_filenames is None:
            f.write('# No filenames were given to export_predictions; this usually indicates\n')
            f.write('# that the sco calculation was run on a dataset imported directly into\n')
            f.write('# Python instead of via filenames.\n')
        else:
            for im in stimulus_image_filenames:
                f.write('%s\n' % im)
    return {'lh_exported_predictions_filename': flnms[0],
            'rh_exported_predictions_filename': flnms[1],
            'exported_image_ordering_filename': ord_flnm}
