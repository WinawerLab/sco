####################################################################################################
# contrast/core.py
# The second-order contrast module of the standard cortical observer.
# By Noah C. Benson

import numpy                 as     np
from   scipy                 import ndimage as ndi

from   skimage.filters       import gabor_kernel

from   types                 import (IntType, LongType)

from   ..core                import calculates

@calculates()
def calc_contrast_default_parameters(gabor_orientations=8):
    '''
    calc_contrast_default_parameters is a calculator that requires no arguments but sets the following
    to a default if they are not provided:
      * gabor_orientations (set to 8 by default)
    If gabor_orientations is a list, then it is taken to be a list of the orientations (in radians),
    and these orientations are used; if it is an integer, then evenly spaced orientations are used:
    pi*k/n where k is in [0, n-1] and n is gabor_orientations.
    '''
    if isinstance(gabor_orientations, (IntType, LongType)):
        gabor_orientations = [np.pi * float(k)/float(gabor_orientations)
                              for k in range(gabor_orientations)]
    elif not hasattr(gabor_orientations, '__iter__'):
        raise ValueError('gabor_orientations must be a list or an integer')
    # That's all this needs to do for now. We want gabor_orientations to be an array, because that
    # makes things easier.
    return {'gabor_orientations': np.asarray(gabor_orientations)}

@calculates('stimulus_contrast_functions',
            d2p='normalized_pixels_per_degree',
            imgs='normalized_stimulus_images',
            orients='gabor_orientations',
            ev='stimulus_edge_value')
def calc_stimulus_contrast_functions(imgs, d2p, orients, ev):
    '''
    calc_stimulus_contrast_functions is a calculator that produces a value named 
    'stimulus_contrast_functions' which a list, each element of which, when given a frequency
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
            filtered = np.sum(filtered_orientations.values(), axis=0)
            filtered.setflags(write=False)
            cache[cpd] = filtered
            return filtered
    def _make_stim_contrast_fn(k):
        return lambda f: _stimulus_contrast_function(k, f)
    return {'stimulus_contrast_functions': [_make_stim_contrast_fn(k) for k in range(len(imgs))]}


@calculates()
def calc_divisive_normalization_functions(stimulus_contrast_functions, Kay2013_normalization_r=1,
                                          Kay2013_normalization_s=.5):
    """
    """
    _divisive_normalization_cache = [{} for i in stimulus_contrast_functions]
    def _divisive_normalization_function(func, cpd):
        cache = _divisive_normalization_cache
        if isinstance(cpd, set):
            return {x: _divisive_normalization_function(k, x) for x in cpd}
        elif hasattr(cpd, '__iter__'):
            return [_divisive_normalization_function(k, x) for x in cpd]
        elif cpd in cache:
            return cache[cpd]
        else:
            contrast_img = func(cpd)
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
    return {'stimulus_contrast_functions'}
