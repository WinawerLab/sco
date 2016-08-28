####################################################################################################
# sco/pRF/core.py
# pRF-related calculations for the standard cortical observer library.
# By Noah C. Benson

import numpy                 as     np
import scipy                 as     sp
from   numbers               import Number
from   pysistence            import make_dict

from   ..core                import (iscalc, calculates, calc_chain)

import os, math, itertools, collections


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

@calculates('pRF_responses', 'pRF_pixel_centers', 'pRF_pixel_sigmas',
            ims='filtered_images', x0s='pRF_centers', sigs='pRF_sizes',
            d2p='normalized_pixels_per_degree',
            sz='normalized_stimulus_size')
def calc_pRF_responses(ims, x0s, sigs, d2p, sz):
    '''
    calc_pRF_responses is a calculator that expects the keys filtered_images, pRF_centers, and
    pRF_sizes from the data pool and provides pRF_responses, which are the energies of the pRFs
    extracted from the images. The size of the output is <imgs> x <pRFs>.
    '''
    im_center = np.asarray(sz, dtype=np.float)*0.5
    x0s_px = [(im_center[0] - x0[1]*d2p, im_center[1] +x0[0]*d2p) for x0 in x0s]
    sigs_px = [d2p*sig for sig in sigs]
    pRF_slices = [[[min(szc, max(0, int(round(x0c + k*sig)))) for k in [-3, 3]]
                   for (x0c, szc) in zip(x0, sz)]
                  for (x0, sig) in zip(x0s_px, sigs_px)]
    return {'pRF_pixel_centers': x0s_px,
            'pRF_pixel_sigmas': sigs_px,
            'pRF_responses': np.asarray(
                [[_extract_pRF(im, x0, sl, sig)
                  for (x0, sl, sig) in zip(x0s_px, pRF_slices, sigs_px)]
                 for im in ims])}


