####################################################################################################
# stimulus/core.py
# The stimulus module of the standard cortical observer; core definitions and checks.
# By Noah C. Benson

import numpy                 as     np
from   scipy                 import ndimage    as ndi
from   scipy.interpolate     import (RectBivariateSpline, interp1d)

from   skimage               import data
from   skimage.util          import img_as_float

from   ..core                import calculates

import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='.*From scipy 0.13.0.*')

@calculates()
def calc_stimulus_default_parameters(stimulus_image_filenames=None,
                                     stimulus_images=None,
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
    if stimulus_image_filenames is None and stimulus_images is None:
        raise ValueError('Either stimulus_image_filenames or stimulus_images must be given!')
    elif stimulus_image_filenames is None:
        n = len(stimulus_images)
    else:
        n = len(stimulus_image_filenames)
    # First, fill out lengths:
    if hasattr(mxe, '__iter__'):
        if len(mxe) != n:
            raise ValueError('len(max_eccentricity) != len(stimulus_images)')
    else:
        mxe = [mxe for i in range(n)]
    if hasattr(d2p, '__iter__'):
        if len(d2p) != n:
            raise ValueError('len(normalized_pixels_per_degree) != len(stimulus_images)')
    else:
        d2p = [d2p for i in range(n)]
    if hasattr(asz, '__iter__'):
        if len(asz) != n:
            raise ValueError('len(normalized_stimulus_aperture) != len(stimulus_images)')
    else:
        asz = [asz for i in range(n)]
    if hasattr(stimulus_edge_value, '__iter__'):
        if len(stimulus_edge_value) != n:
            raise ValueError('len(stimulus_edge_value) != len(stimulus_images)')
    else:
        stimulus_edge_value = [stimulus_edge_value for i in range(n)]
    if hasattr(stimulus_pixels_per_degree, '__iter__'):
        if len(stimulus_pixels_per_degree) != n:
            raise ValueError('len(stimulus_pixels_per_degree) != len(stimulus_images)')
    else:
        stimulus_pixels_per_degree = [stimulus_pixels_per_degree for i in range(n)]
    if hasattr(stimulus_aperture_edge_width, '__iter__'):
        if len(stimulus_aperture_edge_width) != n:
            raise ValueError('len(stimulus_aperture_edge_width) != len(stimulus_images)')
    else:
        stimulus_aperture_edge_width = [stimulus_aperture_edge_width for i in range(n)]
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
    # And return all of them as arrays:
    return {'stimulus_edge_value':          np.asarray(stimulus_edge_value),
            'stimulus_pixels_per_degree':   np.asarray(stimulus_pixels_per_degree),
            'stimulus_aperture_edge_width': np.asarray(stimulus_aperture_edge_width),
            'normalized_stimulus_aperture': np.asarray(asz),
            'normalized_pixels_per_degree': np.asarray(d2p),
            'max_eccentricity':             np.asarray(mxe),
            'stimulus_images':              stimulus_images,
            'stimulus_image_filenames':     stimulus_image_filenames}

@calculates('stimulus_images', filenames='stimulus_image_filenames')
def import_stimulus_images(filenames, stimulus_images=None, stimulus_gamma=None):
    '''
    import_stimulus_images is a calculator that expects the 'stimulus_image_filenames' value and 
    converts this, which it expects to be a list of image filenames (potentially containing 
    pre-loaded image matrices), into a list of images loaded from those filenames. It provides
    this list as 'stimulus_images'.
    The optional argument stimulus_gamma may also be passed to this function; if stimulus_gamma
    is given, then it must be one of:
      * an (n x 2) or (2 x n) matrix such that is equivalent to (potentially after transposition)
        a matrix of (x,y) values where x is the input gamma and y is the corrected gamma.
      * a vector of corrected gamma values; if the vector u is of length n, then this is equivalent
        to passing a matrix in which the y-values are the elements of u and the x-values are evenly
        spaced values that cover the interval [0,1]; accordingly there must be at least 2 elements.
      * a function that accepts a number between 0 and 1 and returns the corrected gamma.
    This function supples the stimulus_gamma back as a function that interpolates the numbers if
    numbers are given instead of a function.
    '''
    # First, setup the stimulus_gamma correction:
    if stimulus_gamma is None:
        stimulus_gamma = lambda x: x
    elif hasattr(stimulus_gamma, '__iter__'):
        vals = np.array(stimulus_gamma)
        if len(vals.shape) > 2:
            raise ValueError('stimulus_gamma must be 1D or 2D array')
        if len(vals.shape) == 1:
            n = float(vals.shape[0] - 1)
            vals = np.asarray([[float(i)/n for i in range(vals.shape[0])], vals])
        # Okay, assume here that vals is nx2 or 2xn
        if vals.shape[1] != 2: vals = vals.T
        # and interpolate these
        stimulus_gamma = interp1d(vals[:,0], vals[:,1], kind='cubic')
    elif not hasattr(stimulus_gamma, '__call__'):
        raise ValueError('Given stimulus_gamma argument has neither iter nor call attribute')
    # Now load the images...
    if stimulus_images is not None:
        ims = [img_as_float(im) for im in stimulus_images]
    else:
        ims = filenames if hasattr(filenames, '__iter__') else [filenames]
        ims = [(img_as_float(data.load(im)) if isinstance(im, basestring) else im) for im in ims]
    ims = [stimulus_gamma(np.mean(im, axis=2) if len(im.shape) > 2 else im)    for im in ims]
    # If the stimulus images are different sizes, this will be an array with
    # dtype=object. Otherwise it will be normal
    return {'stimulus_images': np.asarray(ims), 'stimulus_gamma':  stimulus_gamma}

def image_apply_aperture(im, radius,
                         center=None, fill_value=0.5, edge_width=10, crop=True):
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
    crop      = 2*radius if crop is True else crop
    final_sz  = crop if isinstance(crop, (tuple, list)) else (crop, crop)
    final_sz  = [int(round(x)) for x in final_sz]
    final_im  = np.full(final_sz, fill_value)
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
    f2i = float(2 * radius) / float(final_sz[0])
    image_xy = [(x,y)
                for xy in final_xy
                for (dx,dy) in [(xy[0] - final_center[0], xy[1] - final_center[1])]
                for (x,y) in [(dx*f2i + center[0], dy*f2i + center[1])]]
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
    imgs = [ndi.zoom(im, zoom_ratio, cval=ev) if zoom_ratio != 1 else im
            for (im, d2p, nd2p, ev) in zip(imgs, deg2px, normdeg2px, edge_val)
            for zoom_ratio in [float(nd2p)/float(d2p)]]
    # Then apply the aperture
    imgs = [image_apply_aperture(im, rad, fill_value=ev, edge_width=ew)
            for (im, rad, ev, ew) in zip(imgs, normsz, edge_val, edge_width)]
    # That's it! Make it an array because those are easier and we know every image will be the same size.
    return {'normalized_stimulus_images': np.asarray(imgs)}
