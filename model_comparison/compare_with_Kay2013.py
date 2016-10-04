# compare_with_Kay2013.py
#
# this script runs the SCO model on the bandpassed stimuli from http://kendrickkay.net/socmodel/,
# then exports the parameters so the MATLAB version of the model (from the same url) can be run on
# the same stimuli with the same parameters. We then compare the predicted responses between the
# two models.
#
# By William F. Broderick

import sco
import neuropythy.freesurfer as nfs
import os
import glob
import imghdr
import scipy.io as sio


def main(image_base_path, subject_dir='/mnt/WinawerAcadia/Freesurfer_subjects',
         subject='test-sub'):
    """
    Run python SCO and Matlab SOC on the same batch of images

    Arguments
    ---------

    image_base_path: string. This is assumed to either be a directory, in which case the model will
    be run on every image in the directory, or a .mat file containing the image stimuli, in which
    case they will be loaded in and used.

    subject_dir: string or None. If not None, will add this to neuropythy.freesurfer's subject
    paths

    subject: string. The specific subject to run on.
    """
    if subject_dir and subject_dir not in nfs.subject_paths():
        nfs.add_subject_path(subject_dir)
    if os.path.isdir(image_base_path):
        # Interestingly enough, this works regardless of whether image_base_path ends in os.sep or
        # not.
        stimulus_image_filenames = glob.glob(image_base_path + os.sep + "*")
        # imghdr.what(img) returns something if img is an image file (it returns a stirng
        # specifying whta type of image it is). If it's not an image file, it returns None.
        stimulus_image_filenames = [img for img in stimulus_image_filenames
                                    if imghdr.what(img)]
        kwargs = {'stimulus_image_filenames': stimulus_image_filenames[:2]}
        # and we use the default sco_chain
        sco_chain = sco.sco_chain
    # if it's a .mat file
    elif os.path.splitext(image_base_path)[1] == ".mat":
        # in this case, we assume it's stimuli.mat from
        # http://kendrickkay.net/socmodel/index.html#contentsofstimuli. We want to grab only the
        # 'images' key from this and only images 226 through 260/end (stimulus set 3), since each
        # entry is a single frame and thus easy to handle.
        stimulus_images = sio.loadmat(image_base_path)
        stimulus_images = stimulus_images['images'][0, 225:]
        # in this case, we already have the stimulus images, so we don't need the sco chain to do
        # the importing of them.
        kwargs = {'stimulus_images': stimulus_images[:2]}
        # We need to modify the stimulus chain that's part of sco_chain because we don't need the
        # import_stimulus_images step.
        stim_chain = (
            ('calc_normalized_stimulus', sco.stimulus.core.calc_normalized_stimulus_images),
            ('calc_filters', sco.stimulus.core.calc_gabor_filters),
            ('calc_filtered_images', sco.stimulus.core.calc_filtered_images))
        # This is our modified chain.
        sco_chain = (('calc_anatomy', sco.anatomy.calc_anatomy),
                     # need to call calc_chain on this to make it ready to go.
                     ('calc_stimulus', sco.core.calc_chain(stim_chain)),
                     ('calc_pRF', sco.pRF.calc_pRF),
                     ('calc_normalization', sco.normalization.calc_normalization))
    else:
        raise Exception("Don't know how to handle image_base_path %s, must be directory or .mat "
                        "file" % image_base_path)
    # This prepares the sco_chain, making it a callable object
    sco_chain = sco.core.calc_chain(sco_chain)
    # And this runs it
    results = sco_chain(subject=subject, max_eccentricity=20, **kwargs)
    return results
