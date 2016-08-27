####################################################################################################
# stimulus/core.py
# The stimulus module of the standard cortical observer; core definitions and checks.
# By Noah C. Benson

import numpy                 as     np
import scipy                 as     sp
from   scipy                 import ndimage    as ndi
from   scipy.misc            import imresize

from   skimage               import data
from   skimage.util          import img_as_float
from   skimage.filters       import gabor_kernel

from   neuropythy.immutable  import Immutable
from   numbers               import Number
from   pysistence            import make_dict

import os, math, itertools, collections

import ..core import (iscalc, calculates, calc_chain)

def import_images(expect='stimulus_image_filenames', provide='stimulus_images'):
    '''
    import_images() yeilds a calculator that expects the 'stimulus_image_filenames' value and 
    converts this, which it expects to be a list of image filenames (potentially containing 
    pre-images), into a list of images loaded from those filenames. It provides this list as
    'stimulus_images'.
    The optional arguments expect and provide may be set to change the expected and provided keys.
    '''
    @calculates(provides, filenames=expect)
    def _import_images(filenames):
        ims = filenames if hasattr(filenames, '__iter__') else [filenames]
        ims = [img_as_float(data.load(im) if isinstance(im, basestring) else im) for im in ims]
        ims = [np.mean(im, axis=2) if len(im.shape) > 2 else im for im in ims]
        return {provide: ims}
    return _import_images

def normalize_images(expect='stimulus_images', provide='normalized_stimulus_images'):
    '''
    normalize_images() yields a calculator that expects the 'stimulus_images' key from the
    calculation data pool and provides the value 'normalized_images', which will be a set of
    images normalized to a particular size and resolution.
    The expect and provide parameters may be given to change the input and output varianble names
    for the resulting calculator.
    The following arguments may be provided to the resulting calculator:
      * stimulus_edge_value (0), the value outside the projected stimulus.
      * stimulus_pixels_per_degree (24), the pixels per degree for the stimulus (may be a list)
      * normalized_stimulus_size ((300,300)), a tuple describing the width and depth in pixels
      * normalized_pixels_per_degree (15), the number of pixels per degree in the normralized image
    '''
    @calculates(provide, imgs=expect,
                edge_val='stimulus_edge_value',
                deg2px='stimulus_pixels_per_degree',
                normsz='normalized_stimulus_size',
                normdeg2px='normalized_pixels_per_degree'):
    def _normalize_images(imgs, edge_val=0.0, deg2px=24, normsz=(300,300), normdeg2px=15):
        if not hasattr(edge_val,   '__iter__'): edge_val   = [edge_val   for i in imgs]
        if not hasattr(deg2px,     '__iter__'): deg2px     = [deg2px     for i in imgs]
        if not hasattr(normsz,     '__iter__'): normsz     = [normsz     for i in imgs]
        if not hasattr(normdeg2px, '__iter__'): normdeg2px = [normdeg2px for i in imgs]
        # Zoom each image so that the pixels per degree is right:
        imgs = [ndi.zoom(im, d2p/nd2p, cval=ev)
                for (im, d2p, nd2p, ev) in zip(imgs, deg2px, normdeg2px, edge_val)]
        # Grab out the parts we need
        for (k,im,ev,nsz,nd2p) in zip(range(len(imgs)), imgs, edge_val, normsz, normdeg2px):
            newim = np.zeros(normsz) + ev
            (h,w)     = [min(n, o) for (n,o) in zip(newim.shape, im.shape)]
            (nh0,nw0) = [(s - d)/2 for (s,d) in zip(newim.shape, (h,w))]
            (oh0,ow0) = [(s - d)/2 for (s,d) in zip(im.shape,    (h,w))]
            newim[nh0:(nh0+h), nw0:(nw0+w)] = im[oh0:oh0+h), ow0:(ow0+w)]
            imgs[k] = newim
        # That's it!
        return {provide:                        imgs,
                'stimulus_edge_value':          edge_val,
                'stimulus_pixels_per_degree':   deg2px,
                'normalized_stimulus_size':     normsz,
                'normalized_pixels_per_degree': normdeg2px}
    return _normalize_images

#stimulua_image_calc = calc_chain(import_images(),
#                                 normalize_images(),
#                                 generate_gabor_filters(),
#                                 filter_images())
                                 
