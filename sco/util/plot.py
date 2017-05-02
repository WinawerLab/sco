####################################################################################################
# sco/util/plot.py
# Plotting functions used in the SCO project
# by Noah C. Benson

import numpy             as np
import pyrsistent        as pyr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri    as tri
import pint, pimms, os

def cortical_image(prediction, labels, pRFs, max_eccentricity, image_number=None,
                   visual_area=1, image_size=200, n_stds=1,
                   size_gain=1, method='triangulation', subplot=None):
    '''
    cortical_image(pred, labels, pRFs, maxecc) yields an array of figures that reconstruct the
      cortical images for the given sco results datapool. The image is always sized such that the
      width and height span 2 * max_eccentricity degrees (where max_eccentricity is stored in the
      datapool).

    The following options may be given:
      * visual_area (default: 1) must be 1, 2, or 3 and specifies whether to construct the cortical
        image according to V1, V2, or V3.
      * image_number (default: None) specifies which image to construct a cortical image of. If this
        is given as None, then a list of all images in the datapool is returned; otherwise only the
        figure for the image number specified is returned.
      * image_size (default: 200) specifies the width and height of the image in pixels (only for
        method equal to 'pRF_projection'.
      * n_stds (default: 1) specifies how many standard deviations should be used with each pRF when
        projecting the values from the cortical surface back into the image.
      * size_gain (default: 1) specifies a number to multiply the pRF size by when projecting into
        the image.
      * method (default: 'triangulation') should be either 'triangulation' or 'pRF_projection'. The
        former makes the plot by constructing the triangulation of the pRF centers and filling the
        triangles in while the latter creates an image matrix and projects the pRF predicted
        responses into the relevant pRFs.
    '''
    # if not given an image number, we want to iterate through all images:
    if image_number is None:
        return np.asarray([cortical_image(datapool, visual_area=visual_area, image_number=ii,
                                          image_size=image_size, n_stds=n_stds, method=method,
                                          subplot=subplot)
                           for ii in range(len(datapool['image_array']))])
    if subplot is None:
        plotter = plt
        fig = plotter.figure()
    else:
        plotter = subplot
        fig = subplot
    # Some parameter handling:
    maxecc  = float(pimms.mag(max_eccentricity, 'deg'))
    centers = np.asarray([pimms.mag(p.center, 'deg') for p in pRFs])
    sigs    = np.asarray([pimms.mag(p.radius, 'deg') for p in pRFs])
    z       = prediction[:,image_number]
    (x,y,z,sigs) = np.transpose([(xx,yy,zz,ss)
                                 for ((xx,yy),zz,ss,l) in zip(centers,z,sigs,labels)
                                 if l == visual_area and not np.isnan(zz)])
    if method == 'triangulation':
        plotter.tripcolor(tri.Triangulation(x,y), z, cmap='jet', shading='gouraud')
        plotter.axis('equal')
        plotter.axis('off')
        return fig
    elif method == 'pRF_projection':
        # otherwise, we operate on a single image:    
        img        = np.zeros((image_size, image_size, 2))
        img_center = (float(image_size)*0.5, float(image_size)*0.5)
        img_scale  = (img_center[0]/maxecc, img_center[1]/maxecc)
        for (xx,yy,zz,ss) in zip(x,y,z,sigs):
            ss = ss * img_scale[0] * size_gain
            exp_const = -0.5/(ss * ss)
            row = yy*img_scale[0] + img_center[0]
            col = xx*img_scale[1] + img_center[1]
            if row < 0 or col < 0 or row >= image_size or col >= image_size: continue
            r0 = max([0,          int(round(row - ss))])
            rr = min([image_size, int(round(row + ss))])
            c0 = max([0,          int(round(col - ss))])
            cc = min([image_size, int(round(col + ss))])
            (mesh_xs, mesh_ys) = np.meshgrid(np.asarray(range(c0,cc), dtype=np.float) - col,
                                             np.asarray(range(r0,rr), dtype=np.float) - row)
            gaus = np.exp(exp_const * (mesh_xs**2 + mesh_ys**2))
            img[r0:rr, c0:cc, 0] += zz
            img[r0:rr, c0:cc, 1] += gaus
        img = np.flipud(img[:,:,0] / (img[:,:,1] + (1.0 - img[:,:,1].astype(bool))))
        fig = plotter.figure()
        plotter.imshow(img)
        plotter.axis('equal')
        plotter.axis('off')
        return fig
    else:
        raise ValueError('unrecognized method: %s' % method)
    
def corrcoef_image(pred, meas, labels, pRFs, max_eccentricity,
                   visual_area=1, image_size=200, n_stds=1,
                   size_gain=1, method='triangulation', subplot=None):
    '''
    corrcoef_image(imap) plots an image (using sco.util.cortical_image) of the correlation
      coefficients of the prediction versus the measurements in the given results map from
      the SCO model plan, imap.
    '''
    r = np.zeros(meas.shape[0])
    for (i,(p,t)) in enumerate(zip(pred,meas)):
            try:    r[i] = np.corrcoef(p,t)[0,1]
            except: r[0] = 0.0
    r = np.asarray([r]).T
    f = cortical_image(r, labels, pRFs, max_eccentricity, image_number=0,
                       visual_area=visual_area, image_size=image_size,
                       n_stds=n_stds, size_gain=size_gain, method=method,
                       subplot=subplot)
    subplot = plt if subplot is None else subplot
    subplot.clim((-1,1))
    return f

def report_image(anal):
    '''
    report_image(panam) createa a report image from the prediction analysis panal and returns the
      created figure.
    '''
    
    def analq(**kwargs): return anal[pyr.m(**kwargs)]
    
    eccs = np.unique([kk['eccentricity'] for kk in anal.iterkeys() if 'eccentricity' in kk])
    angs = np.unique([kk['polar_angle'] for kk in anal.iterkeys() if 'polar_angle' in kk])
    n_ec = len(eccs)
    n_pa = len(angs)

    (f,axs) = plt.subplots(4,2, figsize=(12,16))
    for (va,ax) in zip([1,2,3,None],axs):
        if va is None:
            r_ec = np.asarray([analq(eccentricity=e) for e in eccs])
            r_pa = np.asarray([analq(polar_angle=a) for a in angs])
        else:
            r_ec = np.asarray([analq(eccentricity=e, label=va) for e in eccs])
            r_pa = np.asarray([analq(polar_angle=a, label=va) for a in angs])
            e_pa = np.sqrt((1.0 - r_pa**2)/(n_pa - 2.0))
            e_ec = np.sqrt((1.0 - r_ec**2)/(n_ec - 2.0))

        ax[0].errorbar(eccs, r_ec, yerr=e_ec, c='r')
        if va is None: ax[0].set_xlabel('(All Areas) Eccentricity [deg]')
        else:          ax[0].set_xlabel('(V%d) Eccentricity [deg]' % va)
        ax[0].set_ylabel('Pearson\'s r')
        ax[0].set_ylim((-1.0,1.0))
        
        ax[1].errorbar(angs, r_pa, yerr=e_pa, c='r')
        if va is None: ax[1].set_xlabel('(All Areas) Polar Angle [deg]')
        else:          ax[1].set_xlabel('(V%d) Polar Angle [deg]' % va)
        ax[1].set_ylabel('Pearson\'s r')
        ax[1].set_xlim((-180.0,180.0))
        ax[1].set_ylim((-1.0,1.0))

    return f
