####################################################################################################
# sco/pRF/core.py
# pRF-related calculations for the standard cortical observer library.
# By Noah C. Benson

import numpy                 as     np
import scipy.sparse          as     sparse

from   ..core                import calculates


@calculates()
def calc_pRF_default_parameters(pRF_frequency_preference_function=None):
    '''
    calc_pRF_default_parameters is a calculator that chooses default values for the options accepted
    by the pRF calculation chain. Currently, there is exactly one of these:
    pRF_frequency_preference_function. This must be, if provided, a function that accepts three
    arguments: (e, s, l) where e is an eccentricity, s is a pRF size (sigma), and l is a visual area
    label (1, 2, or 3 for V1-V3). The function must return a dictionary whose keys are frequencies
    in cycles per degree and whose values are the weights applied to that particular frequency at
    the given eccentricity, sigma, and visual area label. Note that the s parameter is in units of
    pixels while the e parameter is in units of degrees.
    By default, the function returns a Gaussian set of weights around 1 cycle per sigma.
    '''
    #_default_frequencies = [0.5*2.0**(0.5*float(k)) for k in range(10)]
    _default_frequencies = [0.75, 1.0, 1.2, 1.5, 2.0, 3.0]
    _default_std = 0.3
    def _default_pRF_frequency_preference_function(e, s, l):
        ss = 2.0 / s
        weights = np.asarray([np.exp(-0.5*((k - ss)/_default_std)**2)
                              for k in _default_frequencies])
        wtot = np.sum(weights)
        weights /= wtot
        res = {k:v for (k,v) in zip(_default_frequencies, weights) if v > 0.01}
        wtot = np.sum(res.values())
        return {k: (v/wtot) for (k,v) in res.iteritems()}
    if pRF_frequency_preference_function is None:
        return {'pRF_frequency_preference_function': _default_pRF_frequency_preference_function}
    else:
        return {'pRF_frequency_preference_function': pRF_frequency_preference_function}

@calculates('pRF_pixel_centers', 'pRF_pixel_sizes',
            x0s='pRF_centers', sigs='pRF_sizes',
            d2ps='normalized_pixels_per_degree',
            ims='normalized_stimulus_images')
def calc_pRF_pixel_data(ims, x0s, sigs, d2ps):
    '''
    calc_pRF_pixel_data is a calculator that adds the pRF_pixel_centers and pRF_pixel_sizes to the
    datapool; these are translations of the pRF data into image pixels. The following are required
    inputs for the calculator:
      * pRF_centers
      * pRF_sizes
      * normalized_pixels_per_degree
      * normalized_stimulus_images
    Both of the outputs for this calculator are numpy matrices sized (n,m,...) where n is the number
    of pRFs, m is the number of images, and the ... represents the fact that pRF_pixel_centers is
    size (n,m,2).
    '''
    im_centers = np.asarray([im.shape for im in ims], dtype=np.float)*0.5
    x0s_px = np.asarray(
        [[(im_center[0] - x0[1]*d2p, im_center[1] + x0[0]*d2p)
          for (im_center, d2p) in zip(im_centers, d2ps)]
         for x0 in x0s])
    sigs_px = np.asarray([[d2p*sig for d2p in d2ps] for sig in sigs])
    return {'pRF_pixel_centers': x0s_px,
            'pRF_pixel_sizes': sigs_px}

def _pRF_matrix(imshape, x0, sig, rad):
    '''
    _pRF_matrix(imshape, x0, sig, rad) yields a scipy sparse csr_matrix of size imshape that
    contains Gaussian blob weights that sum to 1 representing the pRF. The weights extend to
    the radius rad.
    '''
    sig = float(sig) * 0.05
    rad = 3.0*sig
    rrng = map(round, [max([x0[0] - rad, 0]), min([x0[0] + rad, imshape[0]])])
    crng = map(round, [max([x0[1] - rad, 0]), min([x0[1] + rad, imshape[1]])])
    rrng = map(int, rrng)
    crng = map(int, crng)
    if any(s[1] - s[0] <= 0 for s in [rrng, crng]):
        raise ValueError('Bad image or std given to _pRF_matrix')
    cnst = -0.5 / (sig*sig)

    mtx = sparse.lil_matrix(imshape)
    (xmsh,ymsh) = np.meshgrid(np.asarray(range(crng[0], crng[1]), dtype=np.float) - x0[1],
                              np.asarray(range(rrng[0], rrng[1]), dtype=np.float) - x0[0])
    mini_mtx = np.exp(cnst * (xmsh**2 + ymsh**2))
    mini_mtx /= mini_mtx.sum()
    mtx[rrng[0]:rrng[1], crng[0]:crng[1]] = mini_mtx
    return mtx.asformat('csr')

    # Old method (slower):
    #mtx = sparse.lil_matrix(imshape)
    #rad2 = rad*rad
    #for r in range(int(np.floor(rrng[0])), int(np.ceil(rrng[1]))):
    #    dr2 = (r - x0[0])**2
    #    if dr2 > rad2: continue
    #    for c in range(int(np.floor(crng[0])), int(np.ceil(crng[1]))):
    #        d2 = dr2 + (c - x0[1])**2
    #        if d2 <= rad2: mtx[r,c] = np.exp(cnst * d2)
    #mtx = mtx / mtx.sum()
    #return mtx.asformat('csr')

@calculates('pRF_matrices')
def calc_pRF_matrices(pRF_pixel_centers, pRF_pixel_sizes, normalized_stimulus_images,
                      pRF_blob_stds=2):
    '''
    calc_pRF_matrices is a calculator that adds to the datapool a set of images the same size as
    the normalized_stimulus_images for each pRF as the new datum 'pRF_matrices'; these matrices
    store Gaussian weights of the pRFs that sum to 1. They are stored as scipy.sparse.csr_matrix
    matrices to conserve space.
    The following data are required:
      * pRF_pixel_centers, pRF_pixel_sizes, pRF_eccentricity, pRF_v123_labels
      * normalized_stimulus_images
    The following datum is optional:
      * pRF_blob_stds (default 2) specifies how many standard deviations should be included in the
        Gaussian blob that defines the pRF.
    The resulting datum, pRF_matrices, is a numpy matrix sized (n,m) where n is the number of pRFs
    and m is the number of stimulus images. Each element is a sparse matrix the same size as the
    relevant normalized stimulus image whose values represent weights on the pixel and add to 1.
    '''
    # okay, we're going to generate a bunch of these; we want to cache them in case some are
    # identical, which is likely if all images are normalized to the same size
    cache = {}
    matrices = np.empty((len(pRF_pixel_centers), len(normalized_stimulus_images)), dtype=np.object)
    for (i, (x0s, sigs)) in enumerate(zip(pRF_pixel_centers, pRF_pixel_sizes)):
        for (j, (im, x0, sig)) in enumerate(zip(normalized_stimulus_images, x0s, sigs)):
            k = tuple(x0) + (sig,) + im.shape
            if k not in cache:
                cache[k] = _pRF_matrix(im.shape, x0, sig, sig*pRF_blob_stds)
            matrices[i,j] = cache[k]
    return {'pRF_matrices': matrices}

@calculates('pRF_frequency_preferences')
def calc_pRF_frequency_preferences(pRF_v123_labels, pRF_eccentricity, pRF_sizes,
                                   pRF_frequency_preference_function):
    '''
    calc_pRF_frequency_preferences is a calculator that produces the frequency preference
    dictionaries for each pRF using the following required data:
      * pRF_v123_labels
      * pRF_eccentricity
      * pRF_sizes
      * pRF_frequency_preference_function
    Note that all of these are added automatically in either the sco.anatomy calculators or by the
    sco.pRF calc_pRF_default_parameters layer of the pRF calculator.
    This calculator puts the datum 'pRF_frequency_preferences' into the datapool; each element of
    this 1D numpy array is a dictionary whose keys are the frequencies and whose values are the
    weights of that frequency for the relevant pRF.
    '''
    prefs = np.asarray(
        [pRF_frequency_preference_function(ecc, sig, lab)
         for (sig, ecc, lab) in zip(pRF_sizes, pRF_eccentricity, pRF_v123_labels)])
    return {'pRF_frequency_preferences': prefs}


