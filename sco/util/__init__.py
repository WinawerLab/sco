####################################################################################################
# sco/util/__init__.py
# Utilities useful in work related to the sco predictions.
# By Noah C. Benson

import numpy             as np
import pyrsistent        as pyr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri    as tri
import pint

from .io import (export_predictions, export_analysis)

# The unit registry that we use
units = pint.UnitRegistry()
# we want to define the pixel unit
units.define('pixel = [image_length] = px')

def lookup_labels(labels, data_by_labels, **kwargs):
    '''
    sco.util.lookup_labels(labels, data_by_labels) yields a list the same size as labels in which
      each element i of the list is equal to data_by_labels[labels[i]].
    
    The option null may additionally be passed to lookup_labels; if null is given, then whenever a
    label value from data is not found in labels, it is instead given the value null; if null is not
    given, then an error is raised in this situation.

    The lookup_labels function expects the labels to be integer or numerical values.
    '''
    res = None
    null = None
    raise_q = True
    if 'null' in kwargs:
        null = kwargs['null']
        raise_q = False
    if len(kwargs) > 1 or (len(kwargs) > 0 and 'null' not in kwargs):
        raise ValueError('Unexpected option given to lookup_labels; only null is accepted')
    if raise_q:
        try:
            res = [data_by_labels[lbl] for lbl in labels]
        except:
            raise ValueError('Not all labels found by lookup_labels and no null given')
    else:
        res = [data_by_labels[lbl] if lbl in data_by_labels else null for lbl in labels]
    return pyr.pvector(res)

def cortical_image(datapool, visual_area=1, image_number=None, image_size=200, n_stds=1,
                   size_gain=1, method='triangulation', subplot=None):
    '''
    cortical_image(datapool) yields an array of figures that reconstruct the cortical images for the
    given sco results datapool. The image is always sized such that the width and height span 
    2 * max_eccentricity degrees (where max_eccentricity is stored in the datapool). The following
    options may be given:
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
                           for ii in range(len(datapool['stimulus_images']))])
    if subplot is None:
        plotter = plt
        fig = plotter.figure()
    else:
        plotter = subplot
        fig = subplot
    if method == 'triangulation':
        # deal with pyplot's interactive mode
        maxecc = float(datapool['max_eccentricity'][image_number])
        labs   = datapool['pRF_v123_labels']
        (x,y)   = datapool['pRF_centers'].T
        z       = datapool['predicted_responses'][image_number]

        (x,y,z) = np.transpose([(xx,yy,zz)
                                for (xx,yy,zz,l) in zip(x,y,z,labs)
                                if l == visual_area])

        plotter.tripcolor(tri.Triangulation(x,y), z, cmap='jet', shading='gouraud')
        plotter.axis('equal')
        plotter.axis('off')
        return fig
    
    elif method == 'pRF_projection':
        # otherwise, we operate on a single image:    
        r      = datapool['predicted_responses']
        maxecc = float(datapool['max_eccentricity'][image_number])
        sigs   = datapool['pRF_sizes']
        labs   = datapool['pRF_v123_labels']
    
        (x,y) = datapool['pRF_centers'].T
        z = r[image_number]
    
        (x,y,z,sigs) = np.transpose([(xx,yy,zz,ss)
                                     for (xx,yy,zz,ss,l) in zip(x,y,z,sigs,labs)
                                     if l == visual_area])
    
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
    

