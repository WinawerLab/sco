####################################################################################################
# sco/util/io.py
# Import/export utilities for the sco project
# by Noah C. Benson

import numpy as np
import pyrsistent as pyr
import os, pimms, sys
import nibabel

def export_anatomical_data(data, sub, anat_ids, hemis, name, output_dir,
                           modality=None, create_dir=False, prefix='', suffix='',
                           null=np.nan, dtype=None):
    '''
    export_anatomical_data(data, sub, anatomical_ids, hemis, name, output_dir) exports the
       prediction data in the given data vector or matrix, out to the given directory in a set of
       files appropriate to the modality; for surfaces, this is an lh.<name>.nii.gz and
       rh.<name>.nii.gz file; for volumes, this is just a <name>.nii.gz file.

    Parameters:
      * data must be a vector of data or a matrix of data whose rows match up to the anatomical ids
      * anatomical_ids must be the voxel indices or vertex ids of the data rows
      * hemis must be a list of 1 or -1 values to indicate LH and RH respectively
      * name should be a name for the file, used as a part of the filename such as lh.<name>.nii.gz
      * output_dir must be the name of the output directory
      * freesurfer_subject: the freesurfer subject object

    Options:
      * create_dir (default: False) if True, will create the directory if it does not exist;
        otherwise raises an exception when the directory does not exist.
      * prefix (default: '') may be a string that is prepended to filenames; prepends are performed
        before other prefixes such as 'lh.<prefix><name><suffix>.nii.gz'.
      * suffix (default: '') may be a string that is appended the the filename; appends are
        performed before the file extension is appended, such as in
        'lh.<prefix><name><suffix>.nii.gz'.
      * modality (default: None) may be specified explicitly as 'surface' or 'volume', but will
        auto-detect the modality based on the anatomical_ids if None is given
      * null (default: nan) specifies the null value that should be written for values not in the
        anatomical_ids; if this is nan and the dtype is an integer value, then 0 is used instead
      * dtype (default: None) specifies explicitly the dtype of the exported array; if None, the
        value np.asarray(data).dtype is used
    '''
    anat_ids = np.asarray(anat_ids)
    if len(anat_ids.shape) == 2 and (anat_ids.shape[0] == 1 or anat_ids.shape[1] == 1):
        anat_ids = anat_ids.flatten()
    if modality is None:
        modality = 'volume' if len(anat_ids.shape) == 2 else 'surface'
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        if create_dir:
            os.mkdirs(output_dir)
        if not (create_dir and os.path.exists(output_dir) and os.path.isdir(output_dir)):
            raise ValueError('Output directory does not exist')
    prefix = '' if prefix is None else prefix
    suffix = '' if suffix is None else suffix
    make_fname = lambda p,ext: os.path.join(output_dir, p + prefix + name + suffix + '.' + ext)
    modality = modality.lower()
    data = np.asarray(data)
    data = np.asarray([data]).T if len(data.shape) == 1 else data
    if len(data) != len(anat_ids):
        raise ValueError('anatomical id (%d) and data (%d) sizesmust match' % (
            len(anat_ids), len(data)))
    if dtype is None: dtype = data.dtype
    if np.issubdtype(dtype, np.int) and not np.isfinite(null): null = 0
    affine = sub.LH.ribbon.affine
    file_names = []
    if modality == 'surface':
        for (hname, hid) in [('lh', 1), ('rh', -1)]:
            hobj = getattr(sub, hname.upper())
            hidcs = np.where(hemis == hid)[0]
            n = hobj.vertex_count
            vol = np.full((1,1,n,data.shape[1]), null, dtype=dtype)
            vol[0,0,anat_ids[hidcs],:] = data[hidcs,:]
            img = nibabel.Nifti1Image(vol, affine)
            fnm = make_fname(hname + '.', 'nii.gz')
            img.to_filename(fnm)
            file_names.append(fnm)
    elif modality == 'volume':
        vol = np.full(sub.LH.ribbon.shape + (data.shape[1],), null, dtype=dtype)
        for (row,(i,j,k)) in zip(anat_ids, data): vol[i,j,k,:] = row
        img = nibabel.Nifti1Image(vol, affine)
        fnm = make_fname('', 'nii.gz')
        img.to_filename(fnm)
        file_names.append(fnm)
    else:
        raise ValueError('unrecognized modality: %s' % modality)
    return file_names

@pimms.calc('exported_predictions_filenames')
def export_predictions(predictions, anatomical_ids, modality, hemispheres, freesurfer_subject,
                       labels, image_names, output_directory,
                       create_directories=False, output_prefix='', output_suffix=''):
    '''
    export_predictions is a calc that exports the prediction data in from the sco calculation, which
       must come from sco.sco_plan(...) or similar; at the least it must contain the data documented
       below. The return value is a set of filenames exported.
    
    Required afferent values:
      * prediction:         the prediction matrix
      * anatomical_ids:     the voxel indices or vertex ids
      * image_names:        the list of image names in order
      * modality:           'surface' or 'volume'
      * labels:             the anatomical labels
      * freesurfer_subject: the freesurfer subject object
      * output_directory:   the directory to which to write the results

    Options:
      * create_directories (default: False) if True, will create the directory if it does not exist;
        otherwise raises an exception when the directory does not exist.
      * output_prefix (default: '') may be a string that is prepended to filenames; prepends are
        performed before other prefixes such as 'lh.<prefix><name><suffix>.nii.gz'.
      * output_suffix (default: '') may be a string that is appended the the filename; appends are
        performed before the file extension is appended, such as in
        'lh.<prefix><name><suffix>.nii.gz'.
    '''
    # output the raw predictions themselves
    fnms = export_anatomical_data(predictions, freesurfer_subject, anatomical_ids, hemispheres,
                                  'prediction', output_directory,
                                  create_dir=create_directories, modality=modality,
                                  prefix=output_prefix, suffix=output_suffix)
    # also the image name list
    imnms_fnm = os.path.join(output_directory, output_prefix + 'images' + output_suffix + '.txt')
    with open(imnms_fnm, 'w') as f:
        for imnm in image_names:
            f.write(imnm + '\n')
    fnms.append(imnms_fnm)
    # that's all this calculator does
    return pyr.pvector(fnms)

@pimms.calc('exported_analysis_filenames')
def export_analysis(prediction_analysis, prediction_analysis_labels, output_directory,
                    create_directories=False, output_prefix='', output_suffix=''):
    '''
    export_evaluations(rmap, output_dir) exports the evaluation data in the given results map, which
       must come from sco.sco_plan(...) or similar; at the least it must contain the data documented
       below. The return value is a set of filenames exported.
    
    Required afferent values:
      * prediction_analysis and prediction_analysis_labels: the analysis of the predicted versus
        ground truth data (from sco.analysis)
      * output_directory, the directory to which to write the results

    Options:
      * create_directories (default: False) if True, will create the directory if it does not exist;
        otherwise raises an exception when the directory does not exist.
      * output_prefix (default: '') may be a string that is prepended to filenames; prepends are
        performed before other prefixes such as 'lh.<prefix><name><suffix>.nii.gz'.
      * output_suffix (default: '') may be a string that is appended the the filename; appends are
        performed before the file extension is appended, such as in
        'lh.<prefix><name><suffix>.nii.gz'.
    '''
    # We basically just make a big CSV data-table:
    if not os.path.exists(output_directory) or not os.path.isdir(output_directory):
        if create_directories:
            os.mkdirs(output_directory)
        if not (create_directories
                and os.path.exists(output_directory) and os.path.isdir(output_directory)):
            raise ValueError('Output directory does not exist')
    output_prefix = '' if output_prefix is None else output_prefix
    output_suffix = '' if output_suffix is None else output_suffix
    # Write out the label-group correlation data
    filename = os.path.join(output_directory, output_prefix + 'analysis' + output_suffix + '.csv')
    headers = sorted(list(set([k for m in prediction_analysis.iterkeys() for k in m.iterkeys()])))
    header = ''
    fmt = ''
    for h in headers:
        header += ',' + h
        fmt += ',%s'
    header[0] = '#'
    header = header + ',n,correlation\n'
    fmt = fmt[1:] + '\n'
    with open(filename, 'w') as f:
        f.write(header)
        for (lbl, rval) in prediction_analysis.iteritems():
            n = len(prediction_analysis_labels[lbl])
            tup = tuple([lbl[k] if k in lbl else 'all' for k in headers])
            tup = tup + (n, rval)
            f.write(fmt % tup)
    return pyr.v(filename)
        
