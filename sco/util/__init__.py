####################################################################################################
# sco/util/__init__.py
# Utilities useful in work related to the sco predictions.
# By Noah C. Benson

import numpy as np

def cortical_image(datapool, visual_area=1, image_number=None, image_size=200, n_stds=1):
    '''
    cortical_image(datapool) yields an array of images (as numpy matrices) whose elements
    reconstruct the cortical images for the given sco results datapool. The image is always sized
    such that the width and height span 2 * max_eccentricity degrees (where max_eccentricity is
    stored in the datapool). The following options may be given:
      * visual_area (default: 1) must be 1, 2, or 3 and specifies whether to construct the cortical
        image according to V1, V2, or V3.
      * image_number (default: None) specifies which image to construct a cortical image of. If this
        is given as None, then a 3D numpy array of all images in the datapool is returned; otherwise
        only the image matrix for the image number specified is returned.
      * image_size (default: 200) specifies the width and height of the image in pixels.
      * n_stds (default: 1) specifies how many standard deviations should be used with each pRF when
        projecting the values from the cortical surface back into the image.
    '''
    # if not given an image number, we want to iterate through all images:
    if image_number is None:
        return np.asarray([cortical_image(datapool, visual_area=visual_area, image_number=ii,
                                          image_size=image_size, n_stds=n_stds)
                           for ii in range(len(datapool['stimulus_images']))])
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
        ss = ss*img_scale[0]
        exp_const = -0.5/(ss*ss)
        row = int(round(yy*img_scale[0] + img_center[0]))
        col = int(round(xx*img_scale[1] + img_center[1]))
        if row < 0 or col < 0 or row >= image_size or col >= image_size: continue
        r0 = max([0,          int(round(row - ss))])
        rr = min([image_size, int(round(row + ss))])
        c0 = max([0,          int(round(col - ss))])
        cc = min([image_size, int(round(col + ss))])
        (mesh_xs, mesh_ys) = np.meshgrid(np.asarray(range(c0,cc), dtype=np.float) - col,
                                         np.asarray(range(r0,rr), dtype=np.float) - row)
        gaus = np.exp(exp_const * np.sqrt(mesh_xs**2 + mesh_ys**2))
        img[r0:rr, c0:cc, 0] += zz
        img[r0:rr, c0:cc, 1] += gaus
    return np.flipud(img[:,:,0] / (img[:,:,1] + (1.0 - img[:,:,1].astype(bool))))
    

