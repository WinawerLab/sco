####################################################################################################
# sco/analysis/core.py
# Core implementation of the analysis module of the sco library
# by Noah C. Benson

import numpy as np
import scipy as sp
import pyrsistent as pyr
import nibabel, nibabel.freesurfer.mghformat as fsmgh
import os, sys, pimms
from sco.util import units

@pimms.calc('ground_truth')
def import_ground_truth(ground_truth_filename, modality, anatomical_ids, hemispheres,
                        freesurfer_subject):
    '''
    import_ground_truth is a calculator that imports ground truth data from a given filename or pair
    of filenames (in the case of surface modalities), and converts them into a matrix of
    ground-truth values that is the same size os the matrix 'predictions' produced by the sco plan.

    Required afferent values:
      * ground_truth_filename must be either the filename or a tuple of (lh_filename, rh_filename) if
        the modality is 'surface' instead of 'volume'
      * modality must be 'surface' or 'volume'
      * anatomical_ids (from sco.anatomy)
      * hemispheres (from sco.anatomy)
      * freesurfer_subject (from sco.antomy)

    Provided efferent values:
      * ground_truth: an (n x m) matrix of the measured values whose rows correspond to the
        anatomical ids and whose columns correspond to the images
    '''
    if ground_truth_filename is None: return None
    gt = None
    if modality == 'surface':
        if len(ground_truth_filename) != 2:
            raise ValueError('ground_truth_filename must be (lhnm, rhnm) when modality is surface')
        if pimms.is_map(ground_truth_filename):
            ground_truth_filename = (ground_truth_filename['lh'], ground_truth_filename['rh'])
        gt = np.asarray([None for _ in anatomical_ids])
        hsz = [None,None]
        for (fnm, hid, hnm, hidx) in zip(ground_truth_filename, [1,-1], ['LH', 'RH'], [0,1]):
            h = getattr(freesurfer_subject, hnm)
            n = h.vertex_count
            if fnm.endswith('.mgh') or fnm.endswith('.mgz'):
                img = fsmgh.load(fnm)
            else:
                img = nibabel.load(fnm)
            vol = np.squeeze(img.dataobj.get_unscaled())
            vol = vol if len(vol) == n else vol.T
            hsz[hidx] = vol.shape
            if len(vol) != n: raise ValueError('number of vertices in %s filename incorrect' % hnm)
            hwhere = np.where(hemispheres == hid)[0]
            gt[hwhere] = vol[anatomical_ids[hwhere]].tolist()
        if not np.array_equal(hsz[0][1:], hsz[1][1:]):
            raise ValueError('(LH,RH) ground-truth dims must be the same: (%d, %d)' % tuple(hsz))
        gt = np.asarray(gt.tolist(), dtype=np.float)
    elif modality == 'volume':
        if ground_truth_filename.endswith('.mgh') or ground_truth_filename.endswith('.mgz'):
            img = fsmgh.load(ground_truth_filename)
        else:
            img = nibabel.load(ground_truth_filename)
        vol = img.dataobj.get_unscaled()
        if not np.array_equal(vol.shape[0:3], freesurfer_subject.LH.ribbon.shape[0:3]):
            raise ValueError('ground-truth volume is not shaped identically to subject ribbon')
        gt = np.asarray([vol[i,j,k,:] for (i,j,k) in anatomical_ids], dtype=np.float)
    else:
        raise ValueError('modality must be either \'surface\' or \'volume\'')
    if gt is not None: gt.setflags(write=False)
    return gt

@pimms.calc('prediction_analysis', 'prediction_analysis_labels')
def calc_prediction_analysis(prediction, ground_truth, labels, hemispheres, pRFs):
    '''
    calc_prediction_analysis is a calculator that takes predictions from the sco model and a
    ground-truth dataset and produces an analysis of the prediction accuracy, ready to be plotted or
    exported.

    Required afferent values:
      * prediction (from sco.contrast)
      * ground_truth (from sco.analysis)
      * labels, hemispheres (from sco.anatomy)
      * pRFs (from sco.pRF)

    Provided efferent values:
      * prediction_analysis: a mapping of analysis results, broken down by label and hemisphere
        as well as eccentricity and polar angle. These analyses are in many cases itables that can be
        easily exported as CSV files.
    '''
    if ground_truth is None: return (None, None)
    # For starters, we just get the correlations between predictions and ground truth:
    r = np.zeros(prediction.shape[0], dtype=np.float)
    for (i,p,g) in zip(range(len(r)), prediction, ground_truth):
        try:    r[i] = np.corrcoef(p,g)[0,1]
        except: r[i] = np.nan
    rok = np.where(np.isfinite(r))[0]
    # Okay, lets get some indexes ready; we're going to continue to build these up then at the end
    # we'll calculate correlations over all of them
    lbls = {pyr.m(label=lbl):np.intersect1d(rok, np.where(labels == lbl)[0]) for lbl in np.unique(labels)}
    unq_ls = lbls.keys()
    lbls[pyr.m(hemi='lh')] = np.intersect1d(rok, np.where(hemispheres == 1)[0])
    lbls[pyr.m(hemi='rh')] = np.intersect1d(rok, np.where(hemispheres == -1)[0])
    for l in unq_ls:
        for h in ['lh', 'rh']:
            lbls[l.set('hemi',h)] = np.intersect1d(lbls[l], lbls[pyr.m(hemi=h)])
    general_lbls = lbls.keys() # save these for later
    # Alright, next we want to calc our own eccentricities and polar angles
    centers = np.asarray([p.center.to(units.degree).m for p in pRFs])
    ecc  = np.sqrt(np.sum(centers**2, axis=1))
    angm = np.arctan2(centers[:,1], centers[:,0])
    angp = 90.0 - 180.0/np.pi * angm
    angp[angp > 180] -= 360.0
    rad  = np.asarray([p.radius.to(units.degree).m for p in pRFs])
    # Okay, now we can do some interesting comparisons; let's sort by polar angle and eccentricity:
    ecc_bins = int(np.median([10, 100, int(np.round(np.power(len(ecc), 0.375)))]))
    c = 100.0 / (ecc_bins + 1)
    ecc_bin_walls = np.percentile(ecc, c*np.asarray(range(ecc_bins + 1), dtype=np.float))
    ecc_bin_walls = [(mn,mx) for (mn,mx) in zip(ecc_bin_walls[:-1], ecc_bin_walls[1:])]
    ecc_bin_idcs = [np.where((ecc >= mn) & (ecc <= mx))[0] for (mn,mx) in ecc_bin_walls]
    ecc_bin_means = np.asarray([np.mean(ecc[ii]) for ii in ecc_bin_idcs])
    for l0 in general_lbls:
        for (mu,ii) in zip(ecc_bin_means, ecc_bin_idcs):
            lbls[l0.set('eccentricity', mu)] = np.intersect1d(lbls[l0], ii)
    for (mu,ii) in zip(ecc_bin_means, ecc_bin_idcs):
        lbls[pyr.m(eccentricity=mu)] = np.intersect1d(lbls[l0], ii)
    ang_bins = int(np.median([10, 100, int(np.round(np.power(len(angp), 0.375)))]))
    c = 100.0 / (ang_bins + 1)
    ang_bin_walls = np.percentile(angp, c*np.asarray(range(ang_bins + 1), dtype=np.float))
    ang_bin_walls = [(mn,mx) for (mn,mx) in zip(ang_bin_walls[:-1], ang_bin_walls[1:])]
    ang_bin_idcs = [np.where((angp >= mn) & (angp <= mx))[0] for (mn,mx) in ang_bin_walls]
    ang_bin_means = np.asarray([np.mean(angp[ii]) for ii in ang_bin_idcs])
    for l0 in general_lbls:
        for (mu,ii) in zip(ang_bin_means, ang_bin_idcs):
            lbls[l0.set('polar_angle', mu)] = np.intersect1d(lbls[l0], ii)
    for (mu,ii) in zip(ang_bin_means, ang_bin_idcs):
        lbls[pyr.m(polar_angle=mu)] = np.intersect1d(lbls[l0], ii)
    # Okay, now we have the labels divide up; let's look at correlations of vertices across images;
    res = {}
    for (lbl, idcs) in lbls.iteritems():
        if len(idcs) == 0: continue
        # first, calculate mean responses across the area
        mu_pred = np.mean(prediction[idcs, :],   axis=0)
        mu_trth = np.mean(ground_truth[idcs, :], axis=0)
        # store the correlation of these arrays
        try:    res[lbl] = np.corrcoef(mu_pred, mu_trth)[0,1]
        except: res[lbl] = np.nan
    # That's it!
    return (pyr.pmap(res), pyr.pmap({k:pyr.pvector(v) for (k,v) in lbls.iteritems()}))
