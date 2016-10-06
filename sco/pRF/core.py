####################################################################################################
# sco/pRF/core.py
# pRF-related calculations for the standard cortical observer library.
# By Noah C. Benson

import numpy                 as     np

from   ..core                import calculates


# This function extracts the entire (weighted) pRF region...
def _extract_pRF(im, x0, sl, sig):
    if any(s[1] - s[0] <= 0 for s in sl): return 0
    xrng = map(float, range(sl[1][0], sl[1][1]))
    yrng = map(float, range(sl[0][0], sl[0][1]))
    cnst = -0.5 / (sig*sig)
    gaus = np.exp(
        cnst * np.sum(
            np.asarray([g - x for (g,x) in zip(np.meshgrid(xrng, yrng), reversed(x0))])**2,
            axis=0))
    mag  = np.sum(gaus, axis=None)
    part = im[sl[0][0]:sl[0][1], sl[1][0]:sl[1][1]]
    return np.sum(part * gaus, axis=None) / mag

@calculates()
def calc_pRF_default_parameters(pRF_frequency_preference_function=None):
    '''
    calc_pRF_default_options is a calculator that chooses default values for the options accepted by
    the pRF calculation chain. Currently, there is exactly one of these:
    pRF_frequency_preference_function. This must be, if provided, a function that accepts three
    arguments: (e, s, l) where e is an eccentricity, s is a pRF size (sigma), and l is a visual area
    label (1, 2, or 3 for V1-V3). The function must return a dictionary whose keys are frequencies
    in cycles per degree and whose values are the weights applied to that particular frequency at
    the given eccentricity, sigma, and visual area label. Note that the s parameter is in units of
    pixels while the e parameter is in units of degrees.
    By default, the function returns a Gaussian set of weights around 1 cycle per sigma.
    '''
    _default_frequencies = [0.5*2.0**(0.5*float(k)) for k in range(10)]
    def _default_pRF_frequency_preference_function(e, s, l):
        weights = np.asarray([np.exp(-0.5*((k - s)/s)**2) for k in _default_frequencies])
        weights /= np.sum(weights)
        return {k:v for (k,v) in zip(_default_frequencies, weights) if v > 0.01}
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

@calculates('pRF_responses')
def calc_pRF_responses(pRF_pixel_centers, pRF_pixel_sizes, pRF_eccentricity, pRF_v123_labels,
                       normalized_stimulus_images,
                       stimulus_contrast_functions,
                       pRF_frequency_preference_function):
    '''
    calc_pRF_responses is a calculator that adds to the datapool the estimated responses of each pRF
    as the element pRF_responses. These responses are estimated by using the weighted mean response
    where the weights come from the pRF_frequency_preference_function.
    The following data are required:
      * pRF_pixel_centers, pRF_pixel_sizes, pRF_eccentricity, pRF_v123_labels
      * normalized_stimulus_images
      * stimulus_contrast_functions
      * pRF_frequency_preference_function
    The resulting datum, pRF_responses, is a numpy matrix sized (n,m) where n is the number of pRFs
    and m is the number of stimulus images.
    '''
    responses = np.asarray(
        [[(0 if (x0[0] < 0 or x0[0] >= im.shape[0] or x0[1] < 0 or x0[1] >= im.shape[1]) else
           np.sum([v*contrast[int(x0[0]), int(x0[1])]
                   for (f,v) in fpref.iteritems()
                   for contrast in [scf(f)]])
           / np.sum(fpref.values()))
          for (im, scf, x0, sig) in zip(normalized_stimulus_images,
                                        stimulus_contrast_functions,
                                        np.round(x0s),
                                        sigs)
          for fpref              in [pRF_frequency_preference_function(ecc, sig, lab)]]
         for (x0s, sigs, ecc, lab) in zip(pRF_pixel_centers,
                                          pRF_pixel_sizes,
                                          pRF_eccentricity,
                                          pRF_v123_labels)])
    return {'pRF_responses': responses}

