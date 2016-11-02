####################################################################################################
# contrast/core.py
# The second-order contrast module of the standard cortical observer.
# By Noah C. Benson

import numpy                 as     np
from   scipy                 import ndimage as ndi

from   skimage.filters       import gabor_kernel

from   types                 import (IntType, LongType)

from   ..core                import calculates
from   ..normalization       import _turn_param_into_array

@calculates()
def calc_contrast_default_parameters(pRF_v123_labels, gabor_orientations=8,
                                     Kay2013_normalization_r=1.0, Kay2013_normalization_s=.5):
    '''
    calc_contrast_default_parameters is a calculator that requires no arguments but sets the following
    to a default if they are not provided:
    * gabor_orientations (set to 8 by default)
    * Kay2013_normalization_r (set to 1 by default)
    * Kay2013_normalization_s (set to .5 by default)
    If gabor_orientations is a list, then it is taken to be a list of the orientations (in radians),
    and these orientations are used; if it is an integer, then evenly spaced orientations are used:
    pi*k/n where k is in [0, n-1] and n is gabor_orientations.

    The Kay2013_normalization parameters can be a single float, a list/array of float, or a
    dictionary with 1, 2 and 3 as its keys and with floats as the values, specifying the values for
    these parameters for voxels in areas V1, V2, and V3. This function will take these values and
    form arrays that correspond to the other voxel-level arrays.
    '''
    if isinstance(gabor_orientations, (IntType, LongType)):
        gabor_orientations = [np.pi * float(k)/float(gabor_orientations)
                              for k in range(gabor_orientations)]
    elif not hasattr(gabor_orientations, '__iter__'):
        raise ValueError('gabor_orientations must be a list or an integer')
    # That's all this needs to do for now. We want gabor_orientations to be an array, because that
    # makes things easier.
    return {'gabor_orientations':      np.asarray(gabor_orientations),
            'Kay2013_normalization_r': _turn_param_into_array(Kay2013_normalization_r, pRF_v123_labels),
            'Kay2013_normalization_s': _turn_param_into_array(Kay2013_normalization_s, pRF_v123_labels)}

@calculates('stimulus_contrast_functions',
            d2p='normalized_pixels_per_degree',
            imgs='normalized_stimulus_images',
            orients='gabor_orientations',
            ev='stimulus_edge_value')
def calc_stimulus_contrast_functions(imgs, d2p, orients, ev):
    '''
    calc_stimulus_contrast_functions is a calculator that produces a value named 
    'stimulus_contrast_functions' which an 1d array, each element of which, when given a frequency
    (in cycles/degree), yields an image that has been transformed from the original
    normalized_stimulus_images to a new image the same size in which each pixel represents the
    contrast energy at that position and at the given frequency.
    The function that is returned automatically caches the results from a call and returns them on
    subsequent calls with the same frequency argument.
    '''
    _stimulus_contrast_cache = [{} for i in imgs]
    # make sure there are lists where there should be
    d2ps = d2p if hasattr(d2p, '__iter__') else [d2p for i in imgs]
    evs  = ev  if hasattr(ev,  '__iter__') else [ev  for i in imgs]
    # Now, the function that is called:
    def _stimulus_contrast_function(k, cpd):
        cache = _stimulus_contrast_cache[k]
        # want to make sure that cpd is a float
        cpd = float(cpd)
        if isinstance(cpd, set):
            return {x: _stimulus_contrast_function(k, x) for x in cpd}
        elif hasattr(cpd, '__iter__'):
            return [_stimulus_contrast_function(k, x) for x in cpd]
        elif cpd in cache:
            return cache[cpd]
        else:
            # switch to cycles per pixel
            im = imgs[k]
            cpp = cpd / d2ps[k]
            c = evs[k]
            kerns = [(kn.real, kn.imag)
                     for th in orients
                     for kn in [gabor_kernel(cpp, theta=th)]]
            # The filtered orientations
            filtered_orientations = {
                th: np.sum([ndi.convolve(im, kern_part, mode='constant', cval=c)**2
                            for kern_part in re_im_kern],
                           axis=0)
                for (th, re_im_kern) in zip(orients, kerns)}
            # now, collapse them down to a single filtered image
            # filtered = np.sum(filtered_orientations.values(), axis=0)
            # filtered_orientations.setflags(write=False)
            cache[cpd] = filtered_orientations
            return filtered_orientations
    def _make_stim_contrast_fn(k):
        """makes the stimulus contrast function for image k

        This is necessary because of some weirdness in python scoping; it's neccessary to make sur
        each function gets the right k.
        """
        return lambda f: _stimulus_contrast_function(k, f)
    return {'stimulus_contrast_functions': np.asarray([_make_stim_contrast_fn(k)
                                                       for k in range(len(imgs))])}


@calculates('normalized_contrast_functions')
def calc_divisive_normalization_functions(stimulus_contrast_functions,
                                          normalized_stimulus_images,
                                          stimulus_edge_value,
                                          normalized_pixels_per_degree,
                                          Kay2013_normalization_r,
                                          Kay2013_normalization_s):
    """
    calc_divisive_normalization_functions is a calculator that produces a value named
    'normalized_contrast_functions', which is a 2d array (voxels by images), each element of which
    is a function which, when given a frequency (in cycles/degree), calls the corresponding
    stimulus_contrast_function, gets the resulting energy image, and then divisively normalizes it
    using the corresponding r and s values for the given voxel.

    Kay2013_normalization_r and Kay2013_normalization_s can be a single value (in which case the
    same value is used for each voxel), an array (in which case each voxel uses the corresponding
    value), or a dictionary with 1, 2, and 3 as the keys (in which case each voxel uses the value
    corresponding to its area, V1, V2, or V3).
    """
    _divisive_normalization_cache = [{} for i in stimulus_contrast_functions]

    def _divisive_normalization_function(func_idx, cpd, vox_id):
        cache = _divisive_normalization_cache[func_idx]
        r     = Kay2013_normalization_r[vox_id]
        s     = Kay2013_normalization_s[vox_id]
        ev    = stimulus_edge_value[func_idx]
        cpp   = cpd / normalized_pixels_per_degree[func_idx]
        if (r, s) not in cache:
            cache[(r, s)] = dict()
        cache = cache[(r, s)]
        if isinstance(cpd, set):
            return {x: _divisive_normalization_function(func_idx, x, vox_id) for x in cpd}
        elif hasattr(cpd, '__iter__'):
            return [_divisive_normalization_function(func_idx, x, vox_id) for x in cpd]
        elif cpd in cache:
            return cache[cpd]
        else:
            func       = stimulus_contrast_functions[func_idx]
            im0        = normalized_stimulus_images[func_idx]
            imsmooth   = ndi.convolve(im0,
                                      np.abs(gabor_kernel(cpp)),
                                      mode='constant',
                                      cval=ev)
            filtered   = func(cpd)
            normalized = np.sum([(v**r)/(s**r + imsmooth**r)
                                 for v in filtered.itervalues()],
                                axis=0)
            normalized.setflags(write=False)
            cache[cpd] = normalized
            return normalized
            # need to talk with Noah about this. I think if I'm understanding this correctly, I
            # would need to just recreate that function, replacing it. Because it sums over all
            # orientations and I need to normalize instead (so instead of np.sum, do something else
            # there. So is the summation across orientations happening there or in pRF?)
            
            # So, change the stimulus_contrast_function to get the energy (as is now, DON'T sum
            # across orientations). So each function will return a list of values, one for each
            # orientation. This will take those in and normalize them (still returning a list of
            # values). Then we'll sum across orientations in pRF or in a separate spatial summation
            # step? Probably in pRF, but double check. And make pRF take
            # normalized_contrast_functions (output from here) as its input instead of stimulus
            # contrast functions (don't want to overwrite). In order to enable modularity,
            # calc_chain has ability to rename variables. Will need to play around with that and
            # add something to README.
    def _make_norm_fn(func_idx, vox_idx):
        """makes the normalized contrast function for function func_idx and voxel vox_idx

        This is necessary because of some weirdness in python scoping; it's neccessary to make sure
        each function gets the right indexes.
        """
        return lambda cpd: _divisive_normalization_function(func_idx, cpd, vox_idx)
    return {'normalized_contrast_functions':
            np.asarray([[_make_norm_fn(func_idx, vox_id)
                         for func_idx, _ in enumerate(stimulus_contrast_functions)]
                        for vox_id, _ in enumerate(Kay2013_normalization_r)])}
