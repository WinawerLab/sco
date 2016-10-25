# core.py
#
# core code for comparing model performances
#
# By William F Broderick

import warnings
import pandas as pd
import numpy as np
from scipy import io as sio
import os

# Keys from the results dict that correspond to model parameters and so that we want to save in the
# output dataframe
MODEL_DF_KEYS = ['pRF_centers', 'pRF_pixel_sizes', 'pRF_hemispheres',
                 'pRF_voxel_indices', 'SOC_responses', 'predicted_responses',
                 'pRF_pixel_centers', 'pRF_eccentricity', 'pRF_v123_labels', 'pRF_sizes',
                 'pRF_polar_angle', 'Kay2013_output_nonlinearity', 'Kay2013_pRF_sigma_slope',
                 'Kay2013_SOC_constant', 'Kay2013_normalization_r', 'Kay2013_normalization_s',
                 'Kay2013_response_gain', 'pRF_frequency_preferences']
    # we currently aren't grabbing pRF_matrices, because I don't know what to do with them.

# Keys from the results dict that correspond to model setup and so we want
# to save them but not in the dataframe.
MODEL_SETUP_KEYS = ['max_eccentricity', 'normalized_pixels_per_degree', 'stimulus_edge_value',
                    'gabor_orientations', 'stimulus_pixels_per_degree', 'subject',
                    'stimulus_contrast_functions', 'normalized_stimulus_aperture',
                    'pRF_frequency_preference_function', 'stimulus_aperture_edge_width',
                    'normalized_contrast_functions', 'pRF_blob_stds']

# Keys that show the stimulus images in some form.
IMAGES_KEYS = ['stimulus_images', 'normalized_stimulus_images', 'stimulus_image_filenames']

# Keys that show brain data in some form.
BRAIN_DATA_KEYS = ['v123_labels_mgh', 'polar_angle_mgh', 'ribbon_mghs', 'eccentricity_mgh']

# Need to go through Kendrick's socmodel.m file (in knkutils/imageprocessing) to figure out what
# parameters he uses there. Actually, it looks like that isn't exactly what we want (shit.)
# because it assumes more similar parameters across voxels than we have. Instead go through the
# fitting example to create the model? Just don't fit any of the parameters and skip to the
# end. And run it a whole bunch of times for all the different voxels.

def _check_default_keys(results):
    default_keys = (MODEL_DF_KEYS + MODEL_SETUP_KEYS + IMAGES_KEYS + BRAIN_DATA_KEYS)
    for k in results:
        if k not in default_keys:
            warnings.warn("Results key %s is not in any of our default key sets!" % k)
    for k in default_keys:
        if k not in results:
            warnings.warn("Default key %s not in results!" % k)

def create_setup_dict(results, keys=None):
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

def create_images_dict(results, keys=None):
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

def create_brain_dict(results, keys=None):
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

def create_model_dataframe(results, image_names, model_df_path="./soc_model_params.csv", keys=None,
                           image_num=None, voxel_num=None):
    """creates the model dataframe from the results dict

    This function takes in a results dict create by one sco_chain run and turns it into a pandas
    dataframe for further analysis or export. The dataframe *does not* contain all of the items in
    results. It only contains those items that are necessary for calculating the output of the
    model. Those keys are given by the constant MODEL_DF_KEYS and can be overwritten by the user
    using the kwarg keys.

    We need a list / array image_names, which gives the names of each of the images, in order. This
    way we can give them more informative labels (even if it's just the indexes in a matlab array,
    as when using Kay2013's stimuli.mat) for direct comparison. the values of image_names can
    either be all integers or all strings, the two cannot mix.

    We also assume that this results dictionary will contain the key 'predicted_responses', which
    is the final output of the model. We assume that this is an array of voxels x images, and so
    infer the number of voxels and images that this model was fit on from this array. If for some
    reason this is not true (which it really should be -- you will basically always have
    predicted_responses), then set image_num and voxel_num to the number of images and voxels,
    respectively.

    Arguments
    =============================

    model_df_path: string. Absolute path to save the model dataframe at. By default, will be saved
    as soc_model_params.csv in the current directory.
    """
    model_df_dict = {}
    if voxel_num and image_num:
        warnings.warn("Using user-set voxel_num and image_num instead of inferring from data")
    elif (voxel_num and not image_num) or (image_num and not voxel_num):
        raise Exception("image_num and voxel_num must be both set or both None, you cannot set"
                        " only one!")
    else:
        image_num, voxel_num = results['predicted_responses'].shape
    if keys:
        model_keys = keys
    else:
        model_keys = MODEL_DF_KEYS
    for k in model_keys:
        try:
            # they should all be arrays, but just in case.
            model_df_dict[k] = np.asarray(results[k])
        except KeyError:
            warnings.warn("Results dict does not contain key %s, skipping" % k)
    # We assume that we have three types of variables here, as defined by their shape:
    #
    # 1. One-dimensional. For these, we simply want their values to be one of the columns of our
    # dataframe. Example: pRF_v123_labels, where each value says whether the corresponding voxel is
    # in V1, V2 or V3.
    #
    # 2. Two-dimensional, voxels on the first dimension, second dimension doesn't correspond to
    # images. For these, we just assume that each column of the array should be a separate column
    # in our dataframe (with its name taken from the name of the variable with a suffix of '_0',
    # '_1' etc ). Example: pRF_centers, where each value gives the x and y coordinates of the
    # center of the voxel's pRF.
    #
    # 3. Two-dimensional, voxels on the first dimension, images on the second or similarly, images
    # on the first and voxels on the second dimensions. For these, we assume these variables give
    # the response (of some kind) to each image of each voxel. Thus each row should be a separate
    # column in our dataframe, suffixed with '_image_1', '_image_2', etc. These will correspond to
    # the names in the image dataframe. Example: predicted response, where each value gives the
    # model's predicted response of that voxel to the given image.
    # 
    # 4. Three-dimensional, with voxels on the first dimension, images on the second, and something
    # else on the third; we give each of those third dimensions a separate column similar to case 2
    # above and each of the second dimensions a separate column similar to case 3. So if the array
    # is voxels x 2 x 2, each voxel will have four columns: _image_0_0, image_0_1, image_1_0, and
    # _image_1_1. Example: pRF_pixel_centers, where each value gives the x and y coordinates of the
    # center of each voxel's pRF in pixels for a given image (since images can have different
    # pixels per degrees, this isn't the same for every image).
    #
    # As we loop through these variables, we split up any two dimensional arrays so that each value
    # in the dictionary is a one-dimensional array, all with length equal to the number of
    # voxels. This will make the creation of the dataframe very easy.

    # We use this tmp_dict to construct the model_df
    tmp_dict = {}
    if isinstance(image_names[0], int):
        # if image_names contains integers, then we use this %04d string formatting
        img_format_string = "%s_image_%04d"
    else:
        # else we just use the values as is.
        img_format_string = "%s_image_%s"
    for k, v in model_df_dict.iteritems():
        if len(v.shape) == 1:
            # This is case 1, and we grab the value as is
            tmp_dict[k] = v
        elif len(v.shape) == 2:
            if v.shape[0] == voxel_num:
                if v.shape[1] != image_num:
                    # then this is case two above.
                    for i in range(v.shape[1]):
                        tmp_dict["%s_dim%s" % (k, i)] = v[:, i]
                else:
                    # then this is case three above
                    for i in range(v.shape[1]):
                        tmp_dict[img_format_string % (k, image_names[i])] = v[:, i]
            elif v.shape[0] == image_num:
                if v.shape[1] != voxel_num:
                    raise Exception("For variable %s, images are on the first dimension but voxels"
                                    " not on the second! (dimensions: %s)" % (k, v.shape))
                # then this is case three above
                for i in range(v.shape[0]):
                    tmp_dict[img_format_string % (k, image_names[i])] = v[i, :]
            else:
                raise Exception("Result variable %s is two dimensional but the first dimension "
                                "doesn't correspond to the number of voxels or images (dimensions:"
                                " %s)!" % (k, v.shape))
        elif len(v.shape) == 3:
            # Here, we assume it's voxels x images x something, like case four above.
            if v.shape[0] == voxel_num:
                for i in range(v.shape[1]):
                    for j in range(v.shape[2]):
                        tmp_dict[(img_format_string + "_dim%s") % (k, image_names[i], j)] = v[:, i, j]
            else:
                raise Exception("Result variable %s is three dimensional but the first dimension "
                                "doesn't correspond to the number of voxels (dimensions: %s)!" %
                                (k, v.shape))
        else:
            raise Exception('Result variable %s is not a one, two, or three dimensional array'
                            ' (shape %s)!' % (k, v.shape))
    model_df = pd.DataFrame(tmp_dict)
    # This isn't necessary, but I find it easier to examine if the columns are sorted
    # alphabetically
    model_df = model_df.reindex_axis(sorted(model_df.columns, key=lambda x: x.lower()), axis=1)
    # I prefer the letters, I think that makes it clearer.
    if 'pRF_hemispheres' in model_df.columns:
        model_df.pRF_hemispheres = model_df.pRF_hemispheres.map({1: 'L', -1: 'R'})
    # Finally, we save model_df as a csv for easy importing / exporting
    model_df.to_csv(model_df_path, index_label='voxel')
    sio.savemat(os.path.splitext(model_df_path)[0] + "_image_names.mat",
                {'image_names': image_names})
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
