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
    _default_frequencies = [0.75 * 2.0**(0.5 * k) for k in range(6)]
    _default_std = 0.5
    def _default_pRF_frequency_preference_function(e, s, l):
        ss = 3.0 / s
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
    ** This function is now depricated; its equivalent is calculated in the PRFSpec class. **
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
    ** This function is now depricated; its equivalent is calculated in the PRFSpec class. **
    _pRF_matrix(imshape, x0, sig, rad) yields a scipy sparse csr_matrix of size imshape that
    contains Gaussian blob weights that sum to 1 representing the pRF. The weights extend to
    the radius rad.
    '''
    sig = float(sig)
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

class PRFSpec(object):
    '''
    The PRFSpec class specifies a pRF size and location such that it can be used to extract a pRF
    from an image. Generally the class is used by a combination of the matrix() method and the
    __call__ method.
    '''
    def __init__(self, center, sigma, n_sigmas=3):
        self.center = center
        self.sigma = sigma
        self.n_sigmas = n_sigmas
    def _params(self, imshape, d2p):
        x0 = np.asarray([imshape[0]*0.5 - self.center[1]*d2p, imshape[1]*0.5 + self.center[0]*d2p])
        sig = float(self.sigma) * d2p
        rad = self.n_sigmas * sig
        rrng0 = (int(np.floor(x0[0] - rad)), int(np.ceil(x0[0] + rad)))
        crng0 = (int(np.floor(x0[1] - rad)), int(np.ceil(x0[1] + rad)))
        rrng = (max([rrng0[0], 0]), min([rrng0[1], imshape[0]]))
        crng = (max([crng0[0], 0]), min([crng0[1], imshape[1]]))
        if any(s[1] - s[0] <= 0 for s in [rrng, crng]):
            raise ValueError('Bad image or std given to PRFSpec._params()')
        return (x0, sig, rad, rrng, crng, rrng0, crng0)
    def _weights(self, x0, sig, rrng, crng):
        cnst = -0.5 / (sig*sig)
        (xmsh,ymsh) = np.meshgrid(np.asarray(range(crng[0], crng[1]), dtype=np.float) - x0[1],
                                  np.asarray(range(rrng[0], rrng[1]), dtype=np.float) - x0[0])
        wmtx = np.exp(cnst * (xmsh**2 + ymsh**2))
        # We can trim off the extras now...
        min_w = np.exp(-0.5 * self.n_sigmas * self.n_sigmas)
        wmtx[wmtx < min_w] = 0.0
        wmtx /= wmtx.sum()
        return wmtx
    def matrix(self, imshape, d2p):
        '''
        prf.matrix(im, d2p) or prf.matrix(im.shape, d2p) both yield a sparse matrix containing the
        weights for the given prf over the given image im (or an image of size im,shape); the d2p
        parameter specifies the number of degrees per pixel for the image.
        '''
        if isinstance(imshape, np.ndarray):
            return self.matrix(imshape.shape)
        (x0, sig, rad, rrng, crng, _, _) = self._params(imshape, d2p)
        mini_mtx = self._weights(x0, sig, rrng, crng)
        mtx = sparse.lil_matrix(imshape)
        mtx[rrng[0]:rrng[1], crng[0]:crng[1]] = mini_mtx
        return mtx.asformat('csr')
    def __call__(self, im, d2p, c=None, edge_value=0):
        '''
        prf(im, d2p) yields a tuple (u, w) in which u and w are equal-length vectors; the vector u
          contains the values found in the pRF while the equivalent vector w contains the matching
          weights for the values in u. The parameter d2p specifies the number of pixels per degree.
        prf(im, d2p, c) yields the weighted second moment about the value c*mu where mu is the
          weighted mean of the values in the pRF over the given image im.
        prf(im, d2p) is equivalent to prf(im, d2p, c=None) or prf(im, d2p, None).

        Note that the parameter im may be either a single image or an image stack (in which case it
        must be size (l, m, n) where l is the number of images and m and n are the rows and columns.
        Additionally, there is an optional parameter edge_value which is, by default, 0 and is
        used outside of the image range. Note that, because this function is generally meant to be
        used with contrast images, 0 is appropriate for the edge_value; it indicates that there is
        no contrast beyond the edge of the stimulus.
        '''
        # first we just grab out the values and weights; to do this we first grab the parameters:
        (x0, sig, rad, rrng, crng, rrng0, crng0) = self._params(im.shape, d2p)
        ## weights are from the _weights method:
        w = self._weights(x0, sig, rrng0, crng0)
        ## values are tricky because they may extend off the end:
        stackq = True if len(im.shape) == 3 else False
        u_im = im[:, rrng[0]:rrng[1], crng[0]:crng[1]] if stackq else \
               im[rrng[0]:rrng[1], crng[0]:crng[1]]
        if rrng[0] == rrng0[0] and rrng[1] == rrng0[1] and \
           crng[0] == crng0[0] and crng[1] == crng0[1]:
            u = u_im
        else:
            rows = rrng[1] - rrng[0]
            cols0 = crng0[1] - crng0[0]
            ev = edge_value
            # we may have extra bits on the top...
            u_r0 = np.full((rrng[0] - rrng0[0], cols0), ev, dtype=np.float) if rrng0[0] < rrng[0] \
                   else None
            # or on the bottom
            u_rr = np.full((rrng0[1] - rrng[1], cols0), ev, dtype=np.float) if rrng0[1] > rrng[1] \
                   else None
            # we might also have extra bits on the sides
            u_c0 = np.full((rows, crng[0] - crng0[0]), ev, dtype=np.float) if crng0[0] < crng[0] \
                   else [[] for r in range(rows)]
            u_cc = np.full((rows, crng0[1] - crng[1]), ev, dtype=np.float) if crng0[1] > crng[1] \
                   else [[] for r in range(rows)]
            # now put them together:
            if stackq:
                u_mid = np.asarray([np.concatenate((u_c0, uu, u_cc), axis=1) for uu in u_im])
                u = np.asarray(
                    [uu if len(row_cat) == 1 else np.concatenate(row_cat, axis=0)
                     for uu in u_mid
                     for row_cat in [tuple([r for r in [u_r0, uu, u_rr] if r is not None])]])
            else:
                u_mid = np.concatenate((u_c0, u_im, u_cc), axis=1)
                row_cat = tuple([r for r in [u_r0, u_mid, u_rr] if r is not None])
                u = u_mid if len(row_cat) == 1 else np.concatenate(row_cat, axis=0)
        # Okay, now u is the correct size and w is the correct size...
        u = np.asarray([uu.flatten() for uu in u]) if stackq else u.flatten()
        w = w.flatten()
        if c is None:
            return (u, w)
        elif stackq:
            return np.asarray([np.dot(w, (uu - c*np.dot(w, uu))) for uu in u])
        else:
            return np.dot(w, (u - c*np.dot(w, u))**2)

@calculates('pRF_views')
def calc_pRF_views(pRF_centers, pRF_sizes, normalized_stimulus_images, stimulus_edge_value,
                   pRF_blob_stds=2):
    '''
    calc_pRF_views is a calculator that adds to the datapool a set of objects of class PRFSpec;
    these objects represent the pRF and can calculate responses or sparse matrices representing
    the weights over an image of the pRF. Generally, if p is a PRFSpec object, then p(im, c) will
    yield the pRF response calculated in Kay et al (2013) with the parameter c; that is, it
    calculates the weighted second moment about c times the weighted mean of the pRF.
    The following data are required:
      * pRF_centers, pRF_sizes
      * normalized_stimulus_images
    The following datum is optional:
      * pRF_blob_stds (default 2) specifies how many standard deviations should be included in the
        Gaussian blob that defines the pRF.
    The resulting datum, pRF_views, is a numpy vector of length n where n is the number of pRFs.
    '''
    return {'pRF_views': np.asarray(
        [PRFSpec(x0, sig, n_sigmas=pRF_blob_stds) for (x0, sig) in zip(pRF_centers, pRF_sizes)])}
    # okay, we're going to generate a bunch of these; we want to cache them in case some are
    # identical, which is likely if all images are normalized to the same size
    #cache = {}
    #matrices = np.empty((len(pRF_pixel_centers), len(normalized_stimulus_images)), dtype=np.object)
    #for (i, (x0s, sigs)) in enumerate(zip(pRF_pixel_centers, pRF_pixel_sizes)):
    #    for (j, (im, x0, sig)) in enumerate(zip(normalized_stimulus_images, x0s, sigs)):
    #        k = tuple(x0) + (sig,) + im.shape
    #        if k not in cache:
    #            cache[k] = _pRF_matrix(im.shape, x0, sig, sig*pRF_blob_stds)
    #        matrices[i,j] = cache[k]
    #return {'pRF_matrices': matrices}

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


