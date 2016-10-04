# core.py
#
# core code for comparing model performances
#
# By William F Broderick

import warnings
import pandas as pd
import numpy as np

# Keys from the results dict that correspond to model parameters and so that we want to save in the
# output dataframe
MODEL_DF_KEYS = ['pRF_centers', 'pRF_pixel_sigmas', 'pRF_hemispheres', 'pRF_responses',
                 'pRF_voxel_indices', 'SOC_normalized_responses', 'predicted_responses',
                 'pRF_pixel_centers', 'pRF_eccentricity', 'pRF_v123_labels', 'pRF_sizes',
                 'pRF_polar_angle', 'Kay2013_output_nonlinearity', 'Kay2013_pRF_sigma_slope',
                 'Kay2013_SOC_constant']

# Keys from the results dict that correspond to model setup and so we want
# to save them but not in the dataframe.
MODEL_SETUP_KEYS = ['filters', 'max_eccentricity', 'min_cycles_per_degree', 'wavelet_frequencies',
                    'normalized_stimulus_size', 'normalized_pixels_per_degree', 'orientations',
                    'stimulus_edge_value', 'gabor_orientations', 'wavelet_steps',
                    'stimulus_pixels_per_degree', 'wavelet_octaves', 'subject']

# Keys that show the stimulus images in some form.
IMAGES_KEYS = ['stimulus_images', 'normalized_stimulus_images', 'filtered_images',
               'stimulus_image_filenames']

# Keys that show brain data in some form.
BRAIN_DATA_KEYS = ['v123_labels_mgh', 'polar_angle_mgh', 'ribbon_mghs', 'eccentricity_mgh']

# Need to go through Kendrick's socmodel.m file (in knkutils/imageprocessing) to figure out what
# parameters he uses there. Actually, it looks like that isn't exactly what we want (shit.)
# because it assumes more similar parameters across voxels than we have. Instead go through the
# fitting example to create the model? Just don't fit any of the parameters and skip to the
# end. And run it a whole bunch of times for all the different voxels.


def create_setup_dataframe(results, keys=None):
    """Return a dictionary containing just the model setup-relevant keys

    I don't think returning a dataframe makes sense here, because we mainly want these around for
    reference. Keys are determined by the constant MODEL_SETUP_KEYS, which can be overwritten by
    the user using the kwarg keys.
    """
    setup_dict = {}
    if keys:
        setup_keys = keys
    else:
        setup_keys = MODEL_SETUP_KEYS
    for k in setup_keys:
        try:
            setup_dict[k] = results[k]
        except KeyError:
            warnings.warn("Results dict does not contain key %s, skipping" % k)
    return setup_dict

def create_images_dataframe(results, keys=None):
    """Return a dictionary containing just the keys corresponding to the stimulus images

    I don't think returning a dataframe makes sense here, because we mainly want these around for
    reference. Keys are determined by the constant IMAGES_KEYS, which can be overwritten by the
    user using the kwarg keys.
    """
    images_dict = {}
    if keys:
        images_keys = keys
    else:
        images_keys = IMAGES_KEYS
    for k in images_keys:
        try:
            images_dict[k] = results[k]
        except KeyError:
            warnings.warn("Results dict does not contain key %s, skipping" % k)
    return images_dict

def create_brain_dataframe(results, keys=None):
    """Return a dictionary containing just the keys corresponding to the input brain images

    I don't think returning a dataframe makes sense here, because we mainly want these around for
    reference. Keys are determined by the constant BRAIN_DATA_KEYS, which can be overwritten by the
    user using the kwarg keys.
    """
    brain_dict = {}
    if keys:
        brain_keys = keys
    else:
        brain_keys = BRAIN_DATA_KEYS
    for k in brain_keys:
        try:
            brain_dict[k] = results[k]
        except KeyError:
            warnings.warn("Results dict does not contain key %s, skipping" % k)
    return brain_dict

def create_model_dataframe(results, model_df_path="./soc_model_params.csv", keys=None,
                           voxel_num_greater_than_images=True, voxel_num=None):
    """creates the model dataframe from the results dict

    This function takes in a results dict create by one sco_chain run and turns it into a pandas
    dataframe for further analysis or export. The dataframe *does not* contain all of the items in
    results. It only contains those items that are necessary for calculating the output of the
    model. Those keys are given by the constant MODEL_DF_KEYS and can be overwritten by the user
    using the kwarg keys.

    We also make the assumption that there are more voxels in the results than there are images. If
    that's not the case, set voxel_num_greater_than_images equal to False and set voxel_num equal
    to the number of voxels in your results.

    Arguments
    =============================

    model_df_path: string. Absolute path to save the model dataframe at. By default, will be saved
    as soc_model_params.csv in the current directory.
    """
    model_df_dict = {}
    if not voxel_num:
        voxel_num = 0
    if keys:
        model_keys = keys
    else:
        model_keys = MODEL_DF_KEYS
    for k in model_keys:
        try:
            # This is temporary fix, list-like entries will soon all be arrays.
            model_df_dict[k] = np.asarray(results[k])
            # figure out what the number of voxels is as we go through.
            if voxel_num_greater_than_images and max(model_df_dict[k].shape) > voxel_num:
                voxel_num = max(model_df_dict[k].shape)
        except KeyError:
            warnings.warn("Results dict does not contain key %s, skipping" % k)
    # We assume that we have three types of variables here, as defined by their shape:
    #
    # 1. One-dimensional. For these, we simply want their values to be one of the columns of our
    # dataframe. Example: pRF_v123_labels, where each value says whether the corresponding voxel is
    # in V1, V2 or V3.
    #
    # 2. Two-dimensional, voxels on the first dimension. For these, we just assume that each column
    # of the array should be a separate column in our dataframe (with its name taken from the name
    # of the variable with a suffix of '_1', '_2' etc ). Example: pRF_centers, where each value
    # gives the x and y coordinates of the center of the voxel's pRF.
    #
    # 3. Two-dimensional, voxels on the second dimension. For these, we assume that images are on
    # the first dimension and so these variables give the response (of some kind) to each image of
    # each voxel. Thus each row should be a separate column in our dataframe, suffixed with
    # '_image_1', '_image_2', etc. These will correspond to the names in the image
    # dataframe. Example: predicted response, where each value gives the model's predicted response
    # of that voxel to the given image.
    #
    # As we loop through these variables, we split up any two dimensional arrays so that each value
    # in the dictionary is a one-dimensional array, all with length equal to the number of
    # voxels. This will make the creation of the dataframe very easy.

    # We use this tmp_dict to construct the model_df
    tmp_dict = {}
    for k, v in model_df_dict.iteritems():
        if len(v.shape) == 1:
            # This is case 1, and we grab the value as is
            tmp_dict[k] = v
        if len(v.shape) == 2:
            if v.shape[0] == voxel_num:
                # then this is case two above.
                for i in range(v.shape[1]):
                    tmp_dict["%s_%s" % (k, i)] = v[:, i]
            elif v.shape[1] == voxel_num:
                # then this is case three above.
                for i in range(v.shape[0]):
                    tmp_dict["%s_image_%s" % (k, i)] = v[i, :]
            else:
                raise Exception("Result variable %s is two dimensional but neither dimension "
                                "corresponds to the number of voxels (dimensions: %s)!" %
                                (k, v.shape))
        # Here, we have no idea what to do and must throw an error
        elif len(v.shape) > 2:
            raise Exception('Result variable %s is not a two dimensional array!' % k)
    model_df = pd.DataFrame(tmp_dict)
    # This isn't necessary, but I find it easier to examine if the columns are sorted
    # alphabetically
    model_df = model_df.reindex_axis(sorted(model_df.columns, key=lambda x: x.lower()), axis=1)
    # I prefer the letters, I think that makes it clearer.
    if 'pRF_hemispheres' in model_df.columns:
        model_df.pRF_hemispheres = model_df.pRF_hemispheres.map({1: 'L', -1: 'R'})
    # Finally, we save model_df as a csv for easy importing / exporting
    model_df.to_csv(model_df_path)
    return model_df

def _get_size(v):
    """returns the number of entries that v has

    this is necessary because some of our variables are lists, some arrays, etc; so we can't rely
    on a simple len() or .shape. This tries them all and returns the relevant one.
    """
    try:
        # this will work for arrays and array-like things (mgh images for example)
        return v.shape
    except AttributeError:
        # then it's probably a list
        try:
            return len(v)
        except TypeError:
            # then it's probably an integer or another single item
            return 1
