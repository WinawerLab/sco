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

from   ..core                import (iscalc, calculates, calc_chain)

import os, math, itertools, collections


def import_images(expect='stimulus_image_filenames', provide='stimulus_images'):
    '''
    import_images() yeilds a calculator that expects the 'stimulus_image_filenames' value and 
    converts this, which it expects to be a list of image filenames (potentially containing 
    pre-images), into a list of images loaded from those filenames. It provides this list as
    'stimulus_images'.
    The optional arguments expect and provide may be set to change the expected and provided keys.
    '''
    @calculates(provide, filenames=expect)
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
                normdeg2px='normalized_pixels_per_degree')
    def _normalize_images(imgs, edge_val=0.0, deg2px=24, normsz=(300,300), normdeg2px=15):
        if not hasattr(edge_val,   '__iter__'): edge_val   = [edge_val   for i in imgs]
        if not hasattr(deg2px,     '__iter__'): deg2px     = [deg2px     for i in imgs]
        # Zoom each image so that the pixels per degree is right:
        imgs = [ndi.zoom(im, float(normdeg2px)/float(d2p), cval=ev)
                for (im, d2p, ev) in zip(imgs, deg2px, edge_val)]
        # Grab out the parts we need
        for (k,im,ev) in zip(range(len(imgs)), imgs, edge_val):
            newim = np.zeros(normsz) + ev
            (h,w)     = [min(n, o) for (n,o) in zip(newim.shape, im.shape)]
            (nh0,nw0) = [(s - d)/2 for (s,d) in zip(newim.shape, (h,w))]
            (oh0,ow0) = [(s - d)/2 for (s,d) in zip(im.shape,    (h,w))]
            newim[nh0:(nh0+h), nw0:(nw0+w)] = im[oh0:(oh0+h), ow0:(ow0+w)]
            imgs[k] = newim
        # That's it!
        return {provide:                        imgs,
                'stimulus_edge_value':          edge_val,
                'stimulus_pixels_per_degree':   deg2px,
                'normalized_stimulus_size':     normsz,
                'normalized_pixels_per_degree': normdeg2px}
    return _normalize_images

def generate_gabor_filters(provide='filters'):
    '''
    generate_gabor_filters() yields a calculator that expects from its data pool the keys
    'wavelet_octaves', 'wavelet_steps', 'normalized_pixels_per_degree', 'gabor_orientations', and
    'min_cycles_per_degree'.
    The octaves and steps determine how many gabors will exist in the pyramid;the orientations
    may be a number for evenly spaced orientations or a list for specific orientations; the
    min_cycles_per_degree is the lowest frequency (per degree) of any of the wavelets.
    The calculator produces an output whose name may be changed via the optional parameter provide;
    by default it is 'filters', and the format is a matrix of (orientation x (octave steps))
    '''
    @calculates(provide, 'wavelet_frequencies',
                octaves='wavelet_octaves', steps='wavelet_steps',
                orientations='gabor_orientations',
                minf='min_cycles_per_degree', d2p='normalized_pixels_per_degree')
    def _generate_gabor_filters(d2p, octaves=4, steps=2, orientations=4, minf=1):
        if not hasattr(orientations, '__iter__'):
            orientations = np.asarray(range(orientations), dtype=np.float) * math.pi / orientations
        wlt_freqs = np.asarray([2.0**(float(q) / float(steps)) for q in range(octaves * steps + 1)])
        wlt_freqs *= float(minf) / float(d2p)
        filters = np.asarray([[gabor_kernel(f, theta=o) for f in wlt_freqs]
                              for o in orientations])
        return {'filters': filters,
                'orientations': orientations,
                'wavelet_frequencies': wlt_freqs}
    return _generate_gabor_filters

def _filt_img(filt, im, cval):
    # Normalize images for better comparison.
    im = im - 0.5
    filt = filt - np.mean(filt)
    cval -= 0.5
    return np.sqrt(ndi.convolve(im, np.real(filt), mode='constant', cval=cval)**2 + 
                   ndi.convolve(im, np.imag(filt), mode='constant', cval=cval)**2)

def filter_images(expect_filters='filters', expect_images='normalized_stimulus_images',
                  provide='filtered_images'):
    '''
    filter_images() yields a calculator that expects from its data pool the keys 'filters' and
    'normalized_stimulus_images' (each of which can be changed by the optional arguments
    expect_filters and expect_images). It provides the result filtered_images, which can be
    changed by the option provide.
    The result is a list of filtered images, whose pixel values are the mean wavelet power across
    the wavelet pyramid provided in filters.
    '''
    @calculates(provide, filts=expect_filters, imgs=expect_images, cval='stimulus_edge_value')
    def _filter_images(filts, imgs, cval=0.0):
        cval = [cval for i in imgs] if isinstance(cval, Number) else cval
        imgs = [np.sum([_filt_img(f, im, c) for ff in filts for f in ff], axis=0)
                for (im,c) in zip(imgs,cval)]
        return {provide: imgs, 'stimulus_edge_value': cval}
    return _filter_images

stimulus_images_calc = calc_chain(import_images(),
                                  normalize_images(),
                                  generate_gabor_filters(),
                                  filter_images())
                                 
