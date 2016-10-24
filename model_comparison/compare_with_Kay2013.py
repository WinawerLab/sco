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
import numpy as np


def main(image_base_path, stimuli_idx, subject='test-sub',
         subject_dir='/home/billbrod/Documents/SCO-test-data/Freesurfer_subjects'):
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
    # if there's just one value and not a list
    if not hasattr(stimuli_idx, '__iter__'):
        stimuli_idx = [stimuli_idx]
    if os.path.isdir(image_base_path):
        # Interestingly enough, this works regardless of whether image_base_path ends in os.sep or
        # not.
        stimulus_image_filenames = glob.glob(image_base_path + os.sep + "*")
        # imghdr.what(img) returns something if img is an image file (it returns a stirng
        # specifying whta type of image it is). If it's not an image file, it returns None.
        stimulus_image_filenames = [img for img in stimulus_image_filenames
                                    if imghdr.what(img)]
        kwargs = {'stimulus_image_filenames': stimulus_image_filenames[stimuli_idx]}
        # and we use the default sco_chain
        sco_chain = sco.sco_chain
    # if it's a .mat file
    elif os.path.splitext(image_base_path)[1] == ".mat":
        # in this case, we assume it's stimuli.mat from
        # http://kendrickkay.net/socmodel/index.html#contentsofstimuli. We want to grab only the
        # 'images' key from this and only images 226 through 260/end (stimulus set 3; note that we
        # have to convert from MATLAB's 1-indexing to python's 0-indexing), since each entry is a
        # single frame and thus easy to handle.
        stimulus_images = sio.loadmat(image_base_path)
        stimulus_images = stimulus_images['images'][0, stimuli_idx]
        # some of the stimuli have multiple frames associated with them; they're just variations on
        # the same image, so we only take one of them to make predictions for.
        stimulus_images = np.asarray([im if len(im.shape)==2 else im[:, :, 0] for im in stimulus_images])
        kwargs = {'stimulus_images': stimulus_images}
        # in this case, we already have the stimulus images, so we don't need the sco chain to do
        # the importing of them.
        # We need to modify the stimulus chain that's part of sco_chain because we don't need the
        # import_stimulus_images step.
        stim_chain = (
            ('calc_stimulus_default_parameters', sco.stimulus.core.calc_stimulus_default_parameters),
            ('calc_normalized_stimulus', sco.stimulus.core.calc_normalized_stimulus_images))
        # This is our modified chain.
        sco_chain = (('calc_anatomy', sco.anatomy.calc_anatomy),
                     # need to call calc_chain on this to make it ready to go.
                     ('calc_stimulus', sco.core.calc_chain(stim_chain)),
                     ('calc_contrast', sco.contrast.calc_contrast),
                     ('calc_pRF', sco.pRF.calc_pRF),
                     ('calc_normalization', sco.normalization.calc_normalization))
    else:
        raise Exception("Don't know how to handle image_base_path %s, must be directory or .mat "
                        "file" % image_base_path)
    # This prepares the sco_chain, making it a callable object
    sco_chain = sco.core.calc_chain(sco_chain)

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
                        pRF_frequency_preference_function=freq_pref, pRF_blob_stds=1, **kwargs)
    return results


# if __name__ == '__main__':
#     import argparse
#     import core
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "image_base_path",
#         help=("string. This is assumed to either be a directory, in which case the model will be "
#               "be run on every image in the directory, or a .mat file containing the image stimuli"
#               ", in which case they will be loaded in and used."))
#     parser.add_argument("subject", help=("string. The specific subject to run on.If unset, will "
#                                          "use test-sub"))
#     parser.add_argument("model_df_path", help=("string. Absolute path to save the model dataframe"
#                                                " at."))
#     parser.add_argument("stimuli_idx", nargs='+', type=int,
#                         help="list of ints. Which indices in the stimuli to run.")
#     parser.add_argument("-s", "--subject_dir", help=("string (optional). If specified, will add to"
#                                                      "neuropythy.freesurfer's subject paths"),
#                         default=None)
#     args = parser.parse_args()
#     if args.subject_dir is not None:
#         results = main(args.image_base_path, np.asarray(args.stimuli_idx), args.subject)
#     else:
#         results = main(args.image_base_path, np.asarray(args.stimuli_idx), args.subject,
#                        args.subject_dir)
#     # for image_names, we use stimuli_idx plus 1, since we use stimuli_idx in python but we'll use
#     # the names in matlab.
#     core.create_model_dataframe(results, np.array(args.stimuli_idx)+1, args.model_df_path)
