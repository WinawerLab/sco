####################################################################################################
# contrast/core.py
# The second-order contrast module of the standard cortical observer.
# By Noah C. Benson

import numpy                 as     np
import scipy                 as     sp
from   scipy                 import ndimage as ndi

from   skimage               import data
from   skimage.filters       import gabor_kernel

from   numbers               import (Number, Integral)
from   types                 import (IntType, LongType)
from   pysistence            import make_dict

from   ..core                import (iscalc, calculates, calc_chain)

import os, math, itertools, collections

@calculates()
def calc_default_contrast_options(gabor_orientations=8):
    '''
    calc_default_contrast_options is a calculator that requires no arguments but sets the following
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
    # That's all this needs to do for now
    return {'gabor_orientations': gabor_orientations}

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
            filtered = np.sum(
                np.abs([ndi.convolve(im, gabor_kernel(cpp, theta=th), mode='constant', cval=c)
                        for th in orients])**2,
                axis=0)
            filtered.setflags(write=False)
            cache[cpd] = filtered
            return filtered
    return {'stimulus_contrast_functions': [(lambda f: _stimulus_contrast_function(k, f))
                                            for k in range(len(imgs))]}
