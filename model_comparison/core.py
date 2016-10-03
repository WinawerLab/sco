# core.py
#
# core code for comparing model performances
#
# By William F Broderick

# Keys from the results dict that correspond to model parameters and so that we want to save in the
# output dataframe
MODEL_DF_KEYS = ['pRF_centers', 'pRF_pixel_sigmas', 'pRF_hemispheres', 'subject', 'pRF_responses',
                 'pRF_voxel_indices', 'SOC_normalized_responses', 'predicted_responses',
                 'pRF_pixel_centers', 'pRF_eccentricity', 'pRF_v123_labels', 'pRF_sizes',
                 'pRF_polar_angle']

# Keys from the results dict that correspond to model setup and so we want
# to save them but not in the dataframe.
MODEL_SETUP_KEYS = ['filters', 'max_eccentricity', 'min_cycles_per_degree', 'wavelet_frequencies',
                    'normalized_stimulus_size', 'normalized_pixels_per_degree', 'orientations',
                    'stimulus_edge_value', 'gabor_orientations', 'wavelet_steps',
                    'stimulus_pixels_per_degree', 'wavelet_octaves']

# Keys that show the stimulus images in some form.
IMAGES_KEYS = ['stimulus_images', 'normalized_stimulus_images', 'filtered_images']

# Keys that show brain data in some form.
BRAIN_DATA_KEYS = ['v123_labels_mgh', 'polar_angle_mgh', 'ribbons_mgh', 'eccentricity_mgh']

# Need to go through Kendrick's socmodel.m file (in knkutils/imageprocessing) to figure out what
# parameters he uses there. Actually, it looks like that isn't exactly what we want (shit.)
# because it assumes more similar parameters across voxels than we have. Instead go through the
# fitting example to create the model? Just don't fit any of the parameters and skip to the
# end. And run it a whole bunch of times for all the different voxels.


def create_dataframe(results):
    """creates a pandas dataframe from the results dict

    This function takes in a results dict create by one sco_chain run and turns it into a pandas
    dataframe for further analysis or export. The dataframe *does not* contain all of the items in
    results. It only contains those items that are necessary for calculating the output of the
    model. Those keys are given by the kwarg model_df_keys and can be overwritten by the user.
    """


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
