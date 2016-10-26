# compare_with_Kay2013.py
#
# this script runs the SCO model on the bandpassed stimuli from http://kendrickkay.net/socmodel/,
# then exports the parameters so the MATLAB version of the model (from the same url) can be run on
# the same stimuli with the same parameters. We then compare the predicted responses between the
# two models.
#
# By William F. Broderick

import os
import glob
import imghdr
import numpy as np
import neuropythy.freesurfer as nfs
import scipy.io as sio

from ..anatomy import core as anatomy_core
from ..stimulus import core as stimulus_core

from ..anatomy import calc_anatomy
from ..stimulus import calc_stimulus
from ..contrast import calc_contrast
from ..pRF import calc_pRF
from ..normalization import calc_normalization
from ..core import calc_chain


def compare_with_Kay2013(image_base_path, stimuli_idx, voxel_idx=None, subject='test-sub',
                         subject_dir='/home/billbrod/Documents/SCO-test-data/Freesurfer_subjects'):
    """Run python SCO and Matlab SOC on the same batch of images

    Arguments
    ---------

    image_base_path: string. This is assumed to either be a directory, in which case the model will
    be run on every image in the directory, or a .mat file containing the image stimuli, in which
    case they will be loaded in and used.

    stimuli_idx: array. Which stimuli from image_base_path we should use. These are assumed
    to be integers that we use as indexes into the stimulus_images (if image_base_path is a .mat
    file) or stimulus_image_filenames (if it's a directory), and we only pass those specified
    images/filenames to the model.

    voxel_idx: array or None, optional. Which voxels to run the model for and create
    predictions. If None or unset, will run the model for all voxels. Else will use the optional
    calculator sco.anatomy.core.calc_voxel_selector to subset the voxels and run only those indices
    correspond to those included in this array.

    subject_dir: string or None. If not None, will add this to neuropythy.freesurfer's subject
    paths

    subject: string. The specific subject to run on.
    """
    if subject_dir is not None and subject_dir not in nfs.subject_paths():
        nfs.add_subject_path(subject_dir)
    # if there's just one value and not a list
    if not hasattr(stimuli_idx, '__iter__'):
        stimuli_idx = [stimuli_idx]
    if voxel_idx is not None:
        if not hasattr(voxel_idx, '__iter__'):
            voxel_idx = [voxel_idx]
        anat_chain = (('import',           anatomy_core.import_benson14_from_freesurfer),
                      ('calc_pRF_centers', anatomy_core.calc_pRFs_from_freesurfer_retinotopy),
                      ('calc_voxel_selector', anatomy_core.calc_voxel_selector),
                      ('calc_anatomy_defualt_parameters', anatomy_core.calc_anatomy_default_parameters),
                      ('calc_pRF_sizes',   anatomy_core.calc_Kay2013_pRF_sizes))
        anat_chain = calc_chain(anat_chain)
        kwargs = {'voxel_idx': voxel_idx}
    else:
        anat_chain = calc_anatomy
        kwargs = {}
    if os.path.isdir(image_base_path):
        # Interestingly enough, this works regardless of whether image_base_path ends in os.sep or
        # not.
        stimulus_image_filenames = glob.glob(image_base_path + os.sep + "*")
        # imghdr.what(img) returns something if img is an image file (it returns a stirng
        # specifying whta type of image it is). If it's not an image file, it returns None.
        stimulus_image_filenames = [img for img in stimulus_image_filenames
                                    if imghdr.what(img)]
        kwargs.update({'stimulus_image_filenames': stimulus_image_filenames[stimuli_idx]})
        # here, stimuli_names can simply be the filenames we're using
        stimuli_names = stimulus_image_filenames[stimuli_idx]
        # and we use the default sco_chain (with the possible exception of anat_chain, see above)
        sco_chain = (('calc_anatomy', anat_chain),
                     ('calc_stimulus', calc_stimulus),
                     ('calc_contrast', calc_contrast),
                     ('calc_pRF', calc_pRF),
                     ('calc_normalization', calc_normalization))
    # if it's a .mat file
    elif os.path.splitext(image_base_path)[1] == ".mat":
        # in this case, we assume it's stimuli.mat from
        # http://kendrickkay.net/socmodel/index.html#contentsofstimuli. We want to grab only the
        # 'images' key from this and only images 226 through 260/end (stimulus set 3; note that we
        # have to convert from MATLAB's 1-indexing to python's 0-indexing), since each entry is a
        # single frame and thus easy to handle.
        stimulus_images = sio.loadmat(image_base_path)
        stimulus_images = stimulus_images['images'][0, stimuli_idx]
        # some of the stimuli have multiple frames associated with them; we want to predict all of
        # them separately, but remember that they were grouped together for later visualization. I
        # can't figure out how to get this loop into a list comprehension, so we'll have to deal
        # with the slight slowdown. stimuli_names is an array we create to keep track of these
        # images. We will return it to the calling code and eventually pass it to create_model_df.
        tmp = []
        stimuli_names = []
        for idx, im in zip(stimuli_idx, stimulus_images):
            if len(im.shape) == 3:
                for i in range(im.shape[2]):
                    tmp.append(im[:, :, i])
                    stimuli_names.append("%04d_sub%02d" % (idx, i))
            else:
                tmp.append(im)
                stimuli_names.append("%04d" % idx)
        stimuli_names = np.asarray(stimuli_names)
        stimulus_images = np.asarray(tmp)
        kwargs.update({'stimulus_images': stimulus_images})
        # in this case, we already have the stimulus images, so we don't need the sco chain to do
        # the importing of them.
        # We need to modify the stimulus chain that's part of sco_chain because we don't need the
        # import_stimulus_images step.
        stim_chain = (
            ('calc_stimulus_default_parameters', stimulus_core.calc_stimulus_default_parameters),
            ('calc_normalized_stimulus', stimulus_core.calc_normalized_stimulus_images))
        # This is our modified chain.
        sco_chain = (('calc_anatomy', anat_chain),
                     # need to call calc_chain on this to make it ready to go.
                     ('calc_stimulus', calc_chain(stim_chain)),
                     ('calc_contrast', calc_contrast),
                     ('calc_pRF', calc_pRF),
                     ('calc_normalization', calc_normalization))
    else:
        raise Exception("Don't know how to handle image_base_path %s, must be directory or .mat "
                        "file" % image_base_path)
    # This prepares the sco_chain, making it a callable object
    sco_chain = calc_chain(sco_chain)

    # in order to handle the fact that the Kay2013 matlab code only deals with spatial orientation
    # of 3 cpd, we have to define a new pRF_frequency_preference_function to replace the default.
    def freq_pref(e, s, l):
        # This takes in the eccentricity, size, and area, but we don't use any of them, since we
        # just want to use 3 cpd and ignore everything else. And this must be floats.
        return {3: 1}
    # And this runs it. To make sure it has the same size as the the images used in Kendrick's
    # code, we set the normalized_stimulus_aperture, normalized_aperture_edge_width, and
    # normalized_pixels_per_degree values. We want our final image to be 90x90, with the edge
    # taking up 10% of the total image (ie 5% of the radius). setting pRF_blob_std to 1 should
    # speed things up
    results = sco_chain(subject=subject, max_eccentricity=2.7, normalized_stimulus_aperture=15*2.727,
                        normalized_pixels_per_degree=15, stimulus_aperture_edge_width=15*(3-2.727),
                        pRF_frequency_preference_function=freq_pref, pRF_blob_stds=2, **kwargs)
    return results, stimuli_names
