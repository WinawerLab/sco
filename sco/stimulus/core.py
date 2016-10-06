####################################################################################################
# stimulus/core.py
# The stimulus module of the standard cortical observer; core definitions and checks.
# By Noah C. Benson

import numpy                 as     np
import scipy                 as     sp
from   scipy                 import ndimage    as ndi
from   scipy.misc            import imresize
from   scipy.interpolate     import RectBivariateSpline

from   skimage               import data
from   skimage.util          import img_as_float
from   skimage.filters       import gabor_kernel

from   neuropythy.immutable  import Immutable
from   numbers               import Number
from   pysistence            import make_dict

from   ..core                import (iscalc, calculates, calc_chain)

import os, math, itertools, collections


@calculates('stimulus_images', filenames='stimulus_image_filenames')
def import_stimulus_images(filenames):
    '''
    import_stimulus_images is a calculator that expects the 'stimulus_image_filenames' value and 
    converts this, which it expects to be a list of image filenames (potentially containing 
    pre-loaded image matrices), into a list of images loaded from those filenames. It provides
    this list as 'stimulus_images'.
    '''
    ims = filenames if hasattr(filenames, '__iter__') else [filenames]
    ims = [img_as_float(data.load(im) if isinstance(im, basestring) else im) for im in ims]
    ims = [np.mean(im, axis=2) if len(im.shape) > 2 else im for im in ims]
    return {'stimulus_images': ims}

def image_apply_aperture(im, radius, center=None, fill_value=0.5, edge_width=10, crop=True):
    '''
    image_apply_aperture(im, rad) yields an image that has been given a circular aperture centered
    at the middle of the image im with the given radius rad in pixels. The following options may be
    given:
      * fill_value (default 0.5) gives the value filled in outside of the aperture
      * crop (default True) indicates whether the image should be cropped after the aperture is
        applied; possible values are a tuple (r,c) indicating the desired size of the resulting
        image; an integer r, equivalent to (r,r); or True, in which case the image is cropped such
        that it just holds the entire aperture (including any smoothed edge).
      * edge_width (default 10) gives the number of pixels over which the aperture should be
        smoothly extended; 0 gives a hard edge, otherwise a half-cosine smoothing is used.
      * center (default None) gives the center of the aperture as a (row, column) value; None uses
        the center of the image.
    '''
    im = np.asarray(im)
    # First, figure out the final image size
    crop = 2*(radius + edge_width) if crop is True else crop
    final_sz = crop if isinstance(crop, (tuple, list)) else (crop, crop)
    final_im = np.full(final_sz, fill_value)
    # figure out the centers
    center       = (0.5*im.shape[0],       0.5*im.shape[1])       if center is None else center
    final_center = (0.5*final_im.shape[0], 0.5*final_im.shape[1])
    # we may have to interpolate pixels, so setup the interpolation; (0,0) in the lower-left:
    interp = RectBivariateSpline(range(im.shape[0]), range(im.shape[1]), im)
    # prepare to interpolate: find the row/col values for the pixels into which we interpolate
    rad2 = radius**2
    erad2 = (radius + edge_width)**2
    final_xy = [(x,y)
                for x in range(final_im.shape[0]) for xx in [(x - final_center[0])**2]
                for y in range(final_im.shape[1]) for yy in [(y - final_center[1])**2]
                if xx + yy <= erad2]
    image_xy = [(x,y)
                for xy in final_xy
                for (dx,dy) in [(xy[0] - final_center[0], xy[1] - final_center[1])]
                for (x,y) in [(dx + center[0], dy + center[1])]]
    final_xy = np.transpose(final_xy)
    image_xy = np.transpose(image_xy)
    # pull the interpolated values out of the interp structure:
    z = interp(image_xy[0], image_xy[1], grid=False)
    # and put these values into the final image
    for ((x,y),z) in zip(final_xy.T, z): final_im[x,y] = z
    # now, take care of the edge
    if edge_width is 0: return final_im
    for r in range(final_im.shape[0]):
        for c in range(final_im.shape[1]):
            r0 = float(r) - final_center[0]
            c0 = float(c) - final_center[1]
            d0 = r0*r0 + c0*c0
            if d0 > rad2 and d0 <= erad2:
                d0 = np.sqrt(d0) - radius
                w = 0.5*(1.0 + np.cos(d0 * np.pi / edge_width))
                final_im[r,c] = w*final_im[r,c] + (1.0 - w)*fill_value
    # That's it!
    return final_im

@calculates()
def calc_stimulus_default_parameters(stimulus_image_filenames,
                                     stimulus_edge_value=0.5,
                                     stimulus_aperture_edge_width=None,
                                     stimulus_pixels_per_degree=24,
                                     normalized_stimulus_aperture=None,
                                     normalized_pixels_per_degree=None,
                                     max_eccentricity=None):
    '''
    calc_stimulus_default_parameters() is a calculator that expects no particular options, but
    fills in several options if not given. These fall into two categories; first, some options
    are given default values if not provided:
      * stimulus_edge_value is set to 0.5
      * stimulus_pixels_per_degree is set to 24 if not provided
      * stimulus_aperture_edge_width is set to normalized_pixels_per_degree
    Other options are dependent on each other:
      * normalized_stimulus_aperture
      * normalized_pixels_per_degree
      * max_eccentricity
    If all three of these are provided (and not None) then they are left as is; if two are given
    then the last is set to obey the equation
     max_eccentricity * normalized_pixels_per_degree == normalized_stimulus_aperture.
    If one or zero of them is given, then the minimum number of following defaults are used, with
    the remaining value filled in as soon as two of the three values has been assigned:
      * max_eccentricity = 12
      * normalized_pixels_per_degree = 15
    Finally, the parameter stimulus_images is required so that all values can be coerced to arrays
    the appropriate size if necessary.
    '''
    mxe = max_eccentricity
    d2p = normalized_pixels_per_degree
    asz = normalized_stimulus_aperture
    n = len(stimulus_image_filenames)
    # First, fill out lengths:
    if hasattr(mxe, '__iter__'):
        if len(mxe) != n:
            raise ValueError('len(max_eccentricity) != len(stimulus_images)')
    else:
        mxe = [mxe for i in stimulus_image_filenames]
    if hasattr(d2p, '__iter__'):
        if len(d2p) != n:
            raise ValueError('len(normalized_pixels_per_degree) != len(stimulus_images)')
    else:
        d2p = [d2p for i in stimulus_image_filenames]
    if hasattr(asz, '__iter__'):
        if len(asz) != n:
            raise ValueError('len(normalized_stimulus_aperture) != len(stimulus_images)')
    else:
        asz = [asz for i in stimulus_image_filenames]
    if hasattr(stimulus_edge_value, '__iter__'):
        if len(stimulus_edge_value) != n:
            raise ValueError('len(stimulus_edge_value) != len(stimulus_images)')
    else:
        stimulus_edge_value = [stimulus_edge_value for i in stimulus_image_filenames]
    if hasattr(stimulus_pixels_per_degree, '__iter__'):
        if len(stimulus_pixels_per_degree) != n:
            raise ValueError('len(stimulus_pixels_per_degree) != len(stimulus_images)')
    else:
        stimulus_pixels_per_degree = [stimulus_pixels_per_degree for i in stimulus_image_filenames]
    if hasattr(stimulus_aperture_edge_width, '__iter__'):
        if len(stimulus_aperture_edge_width) != n:
            raise ValueError('len(stimulus_aperture_edge_width) != len(stimulus_images)')
    else:
        stimulus_aperture_edge_width = [stimulus_aperture_edge_width
                                        for i in stimulus_image_filenames]
    # Now fix the params that depend on each other:
    (mxe, d2p, asz) = [[None if x == 0 else x for x in xx] for xx in [mxe, d2p, asz]]
    mxe = [m     if m is not None           else \
           12    if d is None or a is None  else \
           a/d
           for (m,d,a) in zip(mxe,d2p,asz)]
    d2p = [d     if d is not None           else \
           15    if a is None               else \
           a/m
           for (m,d,a) in zip(mxe,d2p,asz)]
    asz = [a     if a is not None           else \
           m*d
           for (m,d,a) in zip(mxe,d2p,asz)]
    # And fix the aperture edge if needed:
    stimulus_aperture_edge_width = [d if ew is None else ew
                                    for (ew,d) in zip(stimulus_aperture_edge_width,d2p)]
    # And return all of them:
    return {'stimulus_edge_value':          stimulus_edge_value,
            'stimulus_pixels_per_degree':   stimulus_pixels_per_degree,
            'stimulus_aperture_edge_width': stimulus_aperture_edge_width,
            'normalized_stimulus_aperture': asz,
            'normalized_pixels_per_degree': d2p,
            'max_eccentricity':             mxe}

@calculates('normalized_stimulus_images',
            imgs='stimulus_images',
            edge_val='stimulus_edge_value',
            deg2px='stimulus_pixels_per_degree',
            normsz='normalized_stimulus_aperture',
            normdeg2px='normalized_pixels_per_degree',
            edge_width='stimulus_aperture_edge_width')
def calc_normalized_stimulus_images(imgs, edge_val, deg2px, normsz, normdeg2px, edge_width):
    '''
    calc_normalized_stimulus_images is a calculator that expects the 'stimulus_images' key from the
    calculation data pool and provides the value 'normalized_stimulus_images', which will be a set
    of images normalized to a particular size and resolution.
    The following arguments may be provided to the resulting calculator, and must be provided if the
    calc_stimulus_default_parameters calculator does not appear prior to this calculator in a calc
    chain:
      * stimulus_edge_value (0.5), the value outside the projected stimulus.
      * stimulus_pixels_per_degree (24), the pixels per degree for the stimulus (may be a list)
      * normalized_stimulus_aperture (150), the radius (in pixels) of the aperture to apply after
        each image has been normalized
      * normalized_pixels_per_degree (15), the number of pixels per degree in the normralized image
    '''
    # Zoom each image so that the pixels per degree is right:
    imgs = [ndi.zoom(im, round(float(d2p)/float(nd2p)), cval=ev)
            for (im, d2p, nd2p, ev) in zip(imgs, deg2px, normdeg2px, edge_val)]
    # Then apply the aperture
    imgs = [image_apply_aperture(im, rad, fill_value=ev, edge_width=ew)
            for (im, rad, ev, ew) in zip(imgs, normsz, edge_val, edge_width)]
    # That's it!
    return {'normalized_stimulus_images': imgs}




                                 
