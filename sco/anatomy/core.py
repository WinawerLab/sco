####################################################################################################
# anatomy/core.py
# The anatomy module of the standard cortical observer; core definitions and checks.
# By Noah C. Benson

import numpy                 as     np
import neuropythy            as     neuro
import neuropythy.freesurfer as     neurofs
import nibabel.freesurfer    as     fs
import pyrsistent            as     pyr

import pimms, os, sys, warnings

from   neuropythy.commands   import benson14_retinotopy_command
from   ..util                import (lookup_labels, units)

@pimms.calc('freesurfer_subject')
def import_freesurfer_subject(subject):
    '''
    import_freesurfer_subject is a calculator that requires a subject id (subject) and yields a
    Neuropythy FreeSurfer subject object, freesurfer_subject.

      @ subject Must be one of (a) the name of a FreeSurfer subject found on the subject path,
        (b) a path to a FreeSurfer subject directory, or (c) a neuropythy FreeSurfer subject
        object.
    '''
    if isinstance(subject, basestring):
        subject = neuro.freesurfer_subject(subject)
    if not isinstance(subject, neurofs.Subject):
        raise ValueError('Value given for subject is neither a string nor a neuropythy subject')
    return subject

@pimms.calc('polar_angles', 'eccentricities', 'labels', 'hemispheres', 'anatomical_ids')
def import_benson14_from_freesurfer(freesurfer_subject, max_eccentricity,
                                    modality='volume', import_filter=None):
    '''
    import_benson14_from_freesurfer is a calculation that imports (or creates then imports) the
    Benson et al. (2014) template of retinotopy for the subject, whose neuropythy.freesurfer
    Subject object must be provided in the parameter freesurfer_subject. The optional parameter
    modality (default: 'volume') may be either 'volume' or 'surface', and determines if the loaded
    modality is volumetric or surface-based.

    Required afferent parameters:
      @ freesurfer_subject Must be a valid neuropythy.freesurfer.Subject object.
 
    Optional afferent parameters:
      @ modality May be 'volume' or 'surface' to specify the anatomical modality.
      @ max_eccentricity May specifies the maximum eccentricity value to use.
      @ import_filter If specified, may give a function that accepts four parameters:
        f(polar_angle, eccentricity, label, hemi); if this function fails to return True for the 
        appropriate values of a particular vertex/voxel, then that vertex/voxel is not included in
        the prediction.

    Provided efferent values:
      @ polar_angles    Polar angle values for each vertex/voxel.
      @ eccentricities  Eccentricity values for each vertex/voxel.
      @ labels          An integer label 1, 2, or 3 for V1, V2, or V3, one per vertex/voxel.
      @ hemispheres     1 if left, -1 if right for all vertex/voxel.
      @ anatomical_ids  For vertices, the vertex index (in the appropriate hemisphere) for each;
                        for voxels, the (i,j,k) voxel index for each.

    Notes:
      * polar_angles are always given such that a negative polar angle indicates a RH value and a
        positive polar angle inidicates a LH value
      * anatomical_ids is different for surface and volume modalities
      * labels will always be 1, 2, or 3 indicating V1, V2, or V3

    '''
    subject = freesurfer_subject
    if modality.lower() == 'volume':
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
        # The variables are all mgz volumes, so we need to extract the values:
        labels = np.round(np.abs(label_mgz.dataobj.get_unscaled()))
        angles = angle_mgz.dataobj.get_unscaled()
        eccens = eccen_mgz.dataobj.get_unscaled()
        (lrib, rrib) = [r.dataobj.get_unscaled() for r in ribbon_mgzs]
        # Find the voxel indices first:
        coords = np.asarray(np.where(labels.astype(bool))).T
        # Grab the hemispheres; filter down if something isn't in the ribbon
        tmp   = [(1 if lrib[i,j,k] == 1 else -1, (i,j,k))
                 for (i,j,k) in coords
                 if lrib[i,j,k] != 0 or rrib[i,j,k] != 0
                 if eccens[i,j,k] < max_eccentricity]
        hemis  = np.asarray([r[0] for r in tmp], dtype=np.int)
        coords = np.asarray([r[1] for r in tmp], dtype=np.int)
        # Pull out the angle/eccen data
        angs0   = np.asarray([angles[i,j,k] for (i,j,k) in coords])
        angles  = np.pi/180.0 * (90.0 - angs0*hemis)
        eccens  = np.asarray([eccens[i,j,k] for (i,j,k) in coords], dtype=np.float)
        labels  = np.asarray([labels[i,j,k] for (i,j,k) in coords], dtype=np.int)
    elif modality.lower() == 'surface':
        # make sure there are template volume files that match this subject
        lang = os.path.join(subject.directory, 'surf', 'lh.angle_benson14.mgz')
        lecc = os.path.join(subject.directory, 'surf', 'lh.eccen_benson14.mgz')
        llab = os.path.join(subject.directory, 'surf', 'lh.v123roi_benson14.mgz')
        rang = os.path.join(subject.directory, 'surf', 'rh.angle_benson14.mgz')
        recc = os.path.join(subject.directory, 'surf', 'rh.eccen_benson14.mgz')
        rlab = os.path.join(subject.directory, 'surf', 'rh.v123roi_benson14.mgz')
        if not os.path.exists(lang) or not os.path.exists(rang) or \
           not os.path.exists(lecc) or not os.path.exists(recc) or \
           not os.path.exists(llab) or not os.path.exists(rlab):
            # Apply the template first...
            benson14_retinotopy_command(subject.directory)
        if not os.path.exists(lang) or not os.path.exists(rang) or \
           not os.path.exists(lecc) or not os.path.exists(recc) or \
           not os.path.exists(llab) or not os.path.exists(rlab):
            raise ValueError('No anatomical template found/created for subject: ' + lab)
        lang = fs.mghformat.load(lang).dataobj.get_unscaled().flatten()
        lecc = fs.mghformat.load(lecc).dataobj.get_unscaled().flatten()
        llab = fs.mghformat.load(llab).dataobj.get_unscaled().flatten()
        rang = fs.mghformat.load(rang).dataobj.get_unscaled().flatten()
        recc = fs.mghformat.load(recc).dataobj.get_unscaled().flatten()
        rlab = fs.mghformat.load(rlab).dataobj.get_unscaled().flatten()
        (angs0, eccs, labs) = [np.concatenate([ldat, rdat], axis=0)
                               for (ldat,rdat) in zip([lang, lecc, llab], [rang, recc, rlab])]
        idcs  = np.concatenate([range(len(lang)), range(len(rang))], axis=0)
        vals  = np.intersect1d(np.intersect1d(np.where(labs > 0)[0], np.where(labs < 4)[0]),
                               np.where(eccs < max_eccentricity)[0])
        coords  = idcs[vals]
        hemis   = np.concatenate([[1 for a in lang], [-1 for a in rang]], axis=0)[vals]
        angles  = np.pi/180. * (90.0 - angs0[vals]*hemis)
        eccens  = eccs[vals]
        labels  = np.asarray(labs[vals], dtype=np.int)
    else:
        raise ValueError('Option modality must be \'surface\' or \'volume\'')
    # do the filtering and convert to pvectors
    if import_filter is None:
        idcs = range(len(hemis))
    else:
        idcs = [i for (i,(p,e,l,h)) in enumerate(zip(angles, eccens, labels, hemis))
                if import_filter(p,e,l,h)]
    vals = {'polar_angles':   units.degree * angles[idcs],
            'eccentricities': units.degree * eccens[idcs],
            'labels':         labels[idcs],
            'anatomical_ids': coords[idcs],
            'hemispheres':    hemis[idcs]}
    # make sure they're all write-only
    for v in vals.itervalues():
        v.setflags(write=False)
    return vals
        
@pimms.calc('predicted_responses_exported_q')
def export_predicted_responses(predicted_responses,
                               predicted_responses_filename, image_ordering_filename,
                               freesurfer_subject, anatomical_ids, hemispheres, modality,
                               overwrite_files_on_export=True, export_fill_value=0):
    '''
    export_predicted_responses is a calculation that writes out the results of an SCO model instance
    to disk.

    Required afferent parameters:
      @ predicted_responses Must be a dict of keys (image names) to values (predictions); each value
        must be a vector the same size as anatomical_ids.
      @ freesurfer_subject Must be the neuropythy.freesurfer.Subject object.
      @ anatomical_ids Must be the indices into the anatomical data; this is generated by the calc
        object import_benson14_from_freesurfer.
      @ hemispheres Must be a vector that is -1 for each RH and 1 for each LH; this is generated
        by the calc object import_benson14_from_freesurfer.
      @ predicted_responses_filename Must be the name of the file to which to write the predicted
        responses; this should be an mgh or mgz file.
      @ image_ordering_filename Must be the name of the file to which to write the stimulus image
        ordering that is used in the predicted_responses_filename; this will be written as text;
        may be None to indicate that no file should be written.
      @ modality Must be either 'surface' or 'volume'.

    Optional afferent parameters:
      @ overwrite_files_on_export May be set to False to prevent files from being overwritten.
      @ export_fill_value May give the value that is filled into the volume/surface file
        on export for voxels or vertices that are not in the V1/V2/V3 ROIs.

    Efferent values:
      @ predicted_responses_exported_q Will be set to True if the responses were successfully
        exported and to False otherwise.
    
    Notes:
      * If modality is 'surface' then 'lh.' and 'rh.' are prepended to the output volume filename;
        each will be a 1 x 1 x n x d volume where n is the number of vertices in the hemisphere and
        d is the number of images
      * The result, predicted_responses_exported_q, is set to False if the files exist and the
        overwrite_files_on_export is False; all other issues raise errors
      * if any file cannot be written and overwrite_files_on_export is False, then no files are
        written
    '''
    # Before we go crazy, let's see if we can write things out at all:
    prfnm = predicted_response_filename
    if not overwrite_files_on_export:
        if image_ordering_filename is not None and os.path.exists(image_ordering_filename):
            return False
        if modality.lower() == 'volume' and os.path.exists(prfnm):
            return False
        elif modality.lower() == 'surface' and \
             not all(os.path.exists(hstr + prfnm) for hstr in ['lh.', 'rh.']):
            return False
    # Okay, now we can start actually doing work:
    fill = float(export_fill_value)
    imorder = predicted_responses.keys()
    if modality.lower() == 'volume':
        vol0 = freesurfer_subject.LH.ribbon
        vol0dims = vol0.get_data().shape
        preds = np.full(vol0dims + (len(predicted_responses),), fill, dtype=np.float32)
        for (n,imk) in enumerate(imorder):
            for ((i,j,k),val) in zip(anatomical_ids, predicted_responses[imk]):
                preds[i,j,k,n] = val
        hdr = vol0.header.copy()
        hdr.set_data_dtype(np.float32)
        hdr.set_data_shape(preds.shape)
        mgh = fs.mghformat.MGHImage(preds, vol0.affine, hdr, vol0.extra, vol0.file_map)
        mgh.to_filename(prfnm)
    else:
        flnms = []
        for (hemi, hid) in zip([subject.LH, subject.RH], [1, -1]):
            idcs  = np.where(hemispheres == hid)[0]
            preds = np.full((1, 1, hemi.vertex_count, len(idcs)), fill, dtype=np.float32)
            pidcs = np.asarray(anatomical_ids)[idcs]
            for (i,imk) in enumerate(imorder):
                preds[0,0,pidcs,i] = predicted_responses[imk][idcs]
            mgh = fs.mghformat.MGHImage(preds, np.eye(4))
            mgh_flnm = hemi.name.lower() + '.' + prfnm
            mgh.to_filename(mgh_flnm)
    if image_ordering_filename is not None:
        with open(image_ordering_filename, 'w') as f:
            for im in imorder: f.write('%s\n' % im)
    return True
