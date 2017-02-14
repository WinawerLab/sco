# plot/__init__.py
#
# various functions to help plot SCO model performance.
#
# by William F Broderick and Noah Benson

import matplotlib
import itertools
import matplotlib.pyplot as plt
import matplotlib.tri    as tri
import numpy             as np
import seaborn as sns
import pandas as pd
import neuropythy

# pysurfer is optional
try:
    from surfer import Brain
except ImportError, e:
    if e.message != 'No module named surfer':
        raise

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

def _image_plotter(imgs, **kwargs):
    """simple function to plot images from the images object

    kwargs must contain data, a dataframe with only one unique image_idx value, which specifies
    which image to plot
    
    imgs: 3d numpy array containing all images
    """
    ax = plt.gca()
    _ = kwargs.pop('color')
    try:
        data = kwargs.pop('data')
    except KeyError:
        return
    cmap = kwargs.pop('cmap', 'gray')
    if data.image_idx.nunique() > 1:
        raise Exception("Didn't facet correctly, too many images meet this criteria!")
    ax.imshow(imgs[data.image_idx.unique()][0], cmap=cmap, **kwargs)
    
def plot_images(images, image_value, img_restrictions={}, facet_col=None, facet_row=None,
                 cbar=True, **kwargs):
    """plot stimuli in an easy to arrange, easy to parse way
    
    images is the dictionary created by util.metamers.create_image_struct, which has a 'label' key
    containing a DataFrame that labels images by various properties, and other keys corresponding
    to different image types ('stimulus_images' and 'normalized_stimulus_images'), most likely.
    
    image_value: key in images which contains the values to plot (e.g., 'stimulus_images' or
    'normalized_stimulus_images')

    img_restrictions: dictionary, optional. keys must be columns in images['labels'], values are 
                      the value(s) you want to plot

    facet_col, facet_row: strings, optional. how you want to split up the facetgrid

    **kwargs: will be passed to sns.FacetGrid
    """
    tmp_df, facet_col, facet_row = _facet_plot_shape_df(images['labels'], img_restrictions, facet_col,
                                                        facet_row)
    # we just use this to determine the max and min values; we still need the full one to plot
    imgs = images[image_value][tmp_df.image_idx.unique()]
    size = kwargs.pop('size', 2.3)
    margin_titles = kwargs.pop('margin_titles', True)
    try:
        min_val, max_val = imgs.min(), imgs.max()
    except ValueError:
        # if imgs is not a proper 3d array (which happens when some of the images are different
        # shapes, we can't use a simple .min() and .max())
        min_val = min([im.min() for im in imgs])
        max_val = max([im.max() for im in imgs])
    cmap = kwargs.pop('cmap', 'gray')
    with sns.axes_style('whitegrid', {'axes.grid': False, 'axes.linewidth':0.}):
        g = sns.FacetGrid(tmp_df, col=facet_col, row=facet_row, size=size, margin_titles=margin_titles, **kwargs)
        g.map_dataframe(_image_plotter, imgs=images[image_value], vmin=min_val, vmax=max_val, cmap=cmap)
        g.set(yticks=[], xticks=[])
    if cbar:
        make_colorbar(g.fig, cmap, vmin=min_val, vmax=max_val)
    return g    
    
def _cortical_image_plotter(x, y, z, **kwargs):
    plt.tripcolor(matplotlib.tri.Triangulation(x, y), z, shading='gouraud', **kwargs)

def _facet_plot_shape_df(plot_df, restrictions, facet_col, facet_row):
    """handles much of the shaping and restricting of dataframes to plot them using FacetGrids
    """
    tmp_df = plot_df.copy()
    for k,v in restrictions.iteritems():
        if isinstance(v, basestring) or not hasattr(v, '__iter__'):
            v = [v]
        if k in tmp_df.columns:
            tmp_df = tmp_df[(tmp_df[k].isin(v))]
    if facet_col not in tmp_df.columns:
        facet_col = None
    else:
        # this annoying bit will make sure there's not a None in the variables we're facetting on
        tmp_df[facet_col] = tmp_df[facet_col].replace({None: [i for i in tmp_df[facet_col].unique() if i is not None][0]})
    if facet_row not in tmp_df.columns:
        facet_row = None
    else:
        # this annoying bit will make sure there's not a None in the variables we're facetting on
        tmp_df[facet_row] = tmp_df[facet_row].replace({None: [i for i in tmp_df[facet_row].unique() if i is not None][0]})    
    return tmp_df, facet_col, facet_row

def plot_cortical_images(plot_df, plot_restrictions={}, plot_value='predicted_responses',
                         facet_col=None, facet_row=None, xlabels=False, ylabels=False,
                         set_minmax=True, **kwargs):
    """plot the predicted responses on stimuli ("cortical images")

    plot_restrictions: dictionary, optional. keys must be columns in plot_df, values are the
                       value(s) you want to plot

    plot_value: column in plot_df which contains the values to plot, defaults to `predicted_responses`

    facet_col, facet_row: strings, optional. how you want to split up the facetgrid

    xlabels, ylabels: boolean, optional. whether you want to include labels on x and y axes. False
                      by default

    set_minmax: boolean, optional. whether all images should have the same vmin and vmax values
                (True) or whether they should be allowed to have separate vmin and vmax values 
                (False)
    
    **kwargs: will be passed to sns.FacetGrid
    """
    tmp_df, facet_col, facet_row = _facet_plot_shape_df(plot_df, plot_restrictions, facet_col,
                                                        facet_row)
    size = kwargs.pop('size', 2.3)
    margin_titles = kwargs.pop('margin_titles', True)
    aspect = kwargs.pop('aspect', 1.25)    
    if set_minmax:
        min_val, max_val = tmp_df[plot_value].min(), tmp_df[plot_value].max()
    else:
        min_val, max_val = None, None
    cmap = kwargs.pop('cmap', 'gray')
    with sns.axes_style('whitegrid', {'axes.grid': False, 'axes.linewidth':0.}):
        g = sns.FacetGrid(tmp_df, col=facet_col, row=facet_row, size=size, margin_titles=margin_titles, aspect=aspect,
                          **kwargs)
        g.map(_cortical_image_plotter, 'pRF_centers_dim0', 'pRF_centers_dim1', plot_value, 
              vmin=min_val, vmax=max_val, cmap=cmap)
        g.set(yticks=[], xticks=[])
    if not xlabels:
        g.set_xlabels('')
    if not ylabels:
        g.set_ylabels('')
    if set_minmax:
        make_colorbar(g.fig, cmap, vmin=min_val, vmax=max_val)
    for ax in g.fig.axes[:-1]:
        ax.set_aspect('equal')
    return g

def plot_cortical_images_diff(plot_df, diff_vals, plot_restrictions={},
                              plot_value='predicted_responses', facet_col=None, facet_row=None, 
                              xlabels=False, ylabels=False, set_minmax=True, **kwargs):
    """plot the difference between two specified categories of cortical images
    
    note that the two versions you want to plot have to have the same number of rows in the
    dataframe (probably the same voxels, to be safe), so this will almost certainly fail if you're
    trying to compare V1 and V2, for example.

    diff_vals: dictionary, should be of the form {'a': ['b', 'c']}, where a is a column in plot_df
               and b and c are two values of that column. then we will plot the plot_value of b minus
               the plot_value of c as a cortical image

    plot_value: string, optional. column in plot_df which contains the values to plot, 
                defaults to `predicted_responses`

    plot_restrictions: dictionary, optional. keys must be columns in plot_df, values are the
                       value(s) you want to plot

    facet_col, facet_row: strings, optional. how you want to split up the facetgrid

    xlabels, ylabels: boolean, optional. whether you want to include labels on x and y axes. False
                      by default

    set_minmax: boolean, optional. whether all images should have the same vmin and vmax values
                (True) or whether they should be allowed to have separate vmin and vmax values 
                (False)
    
    **kwargs: will be passed to sns.FacetGrid
"""
    tmp_df, facet_col, facet_row = _facet_plot_shape_df(plot_df, plot_restrictions, facet_col,
                                                                 facet_row)
    new_df = tmp_df[tmp_df[diff_vals.keys()[0]]==diff_vals.values()[0][0]]
    new_df['diff'] =tmp_df[tmp_df[diff_vals.keys()[0]]==diff_vals.values()[0][0]][plot_value] - tmp_df[tmp_df[diff_vals.keys()[0]]==diff_vals.values()[0][1]][plot_value]    
    size = kwargs.pop('size', 2.3)
    margin_titles = kwargs.pop('margin_titles', True)
    aspect = kwargs.pop('aspect', 1.25)    
    if set_minmax:
        min_val, max_val = new_df['diff'].min(), new_df['diff'].max()
    else:
        min_val, max_val = None, None
    cmap = kwargs.pop('cmap', 'RdBu')
    if min_val < 0 and max_val > 0:
        norm = MidpointNormalize(min_val, max_val, 0.)
    else:
        norm = matplotlib.colors.Normalize(min_val, max_val)
    with sns.axes_style('whitegrid', {'axes.grid': False, 'axes.linewidth':0.}):
        g = sns.FacetGrid(new_df, col=facet_col, row=facet_row, size=size, margin_titles=margin_titles, aspect=aspect,
                          **kwargs)
        g.map(_cortical_image_plotter, 'pRF_centers_dim0', 'pRF_centers_dim1', 'diff', 
              vmin=min_val, vmax=max_val, cmap=cmap, norm=norm)
        g.set(yticks=[], xticks=[])
    if not xlabels:
        g.set_xlabels('')
    if not ylabels:
        g.set_ylabels('')
    if set_minmax:
        make_colorbar(g.fig, cmap, vmin=min_val, vmax=max_val)
    for ax in g.fig.axes[:-1]:
        ax.set_aspect('equal')
    return g

def _hemi_check(hemi, mode='pandas'):
    """make sure the hemi string is correctly formatted
    
    mode: {'pandas', 'surfer'}, determines whether our response should be 'R' / 'L' (pandas) or
    'rh' / 'lh' (surfer), based on the conventions of SCO's `model_df` and pysurfer,
    respectively.
    """
    try:
        if mode=='pandas':
            hemi_dict = {'R':'R', 'RH': 'R', 'L': 'L', 'LH': 'L'}
        elif mode=='surfer':
            hemi_dict = {'R':'rh', 'RH': 'rh', 'L': 'lh', 'LH': 'lh'}            
        return hemi_dict[hemi.upper()]
    except KeyError:
        raise Exception("Don't know what to do with hemi %s" % hemi)

def _flat_cortex_pivot_table(hemi, mesh, model_df, df_to_pivot, piv_val='predicted_responses',
                             piv_idx=['model', 'image_type']):
    """
    This is a separate function because we only need to run it once per hemisphere.
    
    hemi: the hemisphere to create the pivot table for, either 'L'/'LH'/'l'/'lh' or 'R'/'RH'/'r'/'rh'

    mesh: a neuropythy projected mesh, constructed by calling e.g., 
          subject.LH.projection(method='equirectangular')

    df_to_pivot: "long form" dataframe containing the values you want to map onto the cortex. Probably 
                 plot_overall_df or SNR_overall_df

    piv_val: the name of the column in df_to_pivot that you want to use as values (and so will be plotted)
             on the cortex

    piv_idx: str or list of strs, the name(s) of the column(s) in df_to_pivot you want to use as indexes in
             the pivot table. These will probably be something like 'model' or 'image_type'
    """
    hemi = _hemi_check(hemi)
    hemi_df = model_df[model_df['pRF_vertex_indices'].isin(mesh.vertex_labels) & model_df['pRF_hemispheres'].isin([hemi])]
    piv = df_to_pivot[df_to_pivot['voxel'].isin(hemi_df['voxel'])]
    piv = piv.pivot_table(values=piv_val, columns='voxel', index=piv_idx, aggfunc=np.mean)
    return hemi_df, piv

def _flat_cortex_shape_data(mesh, piv, piv_idx_vals, hemi_df):
    """
    based on export_predicted_response_surface, need to get this in right shape for plotting

    mesh: a neuropythy projected mesh, constructed by calling e.g., 
          subject.LH.projection(method='equirectangular')

    piv: a pivot table (as created by _flat_cortex_pivot_table) containing the values you want to
         shape
         
    piv_idx_vals: str or tuple of strings, the specific values you want to grab from piv.
    """
    preds = np.full((mesh.coordinates.shape[1]), 0., dtype=np.float32)
    idx = np.array([np.where(mesh.vertex_labels==i)[0] for i in hemi_df['pRF_vertex_indices'].values]).flatten()
    preds[idx] = piv.loc[piv_idx_vals].values
    return preds

class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def _flat_cortex_make_cmap(mesh, plot_val, vmin=None, vmax=None,
                          v123_cmaps = {1: 'Reds', 2: 'Reds', 3: 'Reds'}):

    """This will only plot the corresponding values on V1, V2, and V3 (as specified by mesh property 

    'benson14_visual_area'). The other parts of the cortex will just show the sulci / gyri pattern.

    mesh: a neuropythy projected mesh, constructed by calling e.g., 
          subject.LH.projection(method='equirectangular')

    plot_val: numpy array or string. if an array: must be the same shape as mesh.coordinates for 
               the corresponding hemisphere, and contains the values you wish to make the colormap 
               for. If a string: the name of a neuropythy mesh property that you wish to make the 
               colormap for

    v123_cmaps: dictionary, optional. Keys must be 1, 2, and 3 and values are the colormaps to use
                for V1, V2, and V3.  by default, all three will be matplotlib.cm.Reds.
    """

    for k, v in v123_cmaps.iteritems():
        if isinstance(v, basestring):
            v123_cmaps[k] = plt.get_cmap(v)
    
    def _get_color_vals(properties_dict):
        if properties_dict['curvature'] > 0:
            blend_col = np.asarray(sns.xkcd_palette(['dark grey'])[0])
        else:
            blend_col = np.asarray(sns.xkcd_palette(['light grey'])[0])
        if properties_dict['benson14_visual_area'] == 0 or properties_dict[prop_name] == 0:
            return np.concatenate((99*blend_col/100, [.1]))
        else:
            return np.asarray(v123_cmaps[properties_dict['benson14_visual_area']](norm(properties_dict[prop_name])))
    
    if isinstance(plot_val, basestring):
        if vmin is None:
            vmin = float(mesh.prop(plot_val).min())
        if vmax is None:
            vmax = float(mesh.prop(plot_val).max())
        nuniq = pd.Series(mesh.prop(plot_val)).nunique()
        merge_dict = {}
        prop_name = plot_val
    else:
        if vmin is None:
            vmin = float(plot_val.min())
        if vmax is None:
            vmax = float(plot_val.max())
        nuniq = pd.Series(plot_val).nunique()
        merge_dict = {'summary_vals': plot_val}
        prop_name = 'summary_vals'

    if vmin < 0 and vmax > 0:
        norm = MidpointNormalize(vmin, vmax, 0.)
    else:
        norm = matplotlib.colors.Normalize(vmin, vmax)
        
    map_colors = np.asarray(mesh.map_vertices(_get_color_vals, merge=merge_dict))
    cmap=matplotlib.colors.ListedColormap(map_colors)

    return map_colors, cmap, nuniq, v123_cmaps[1]

def flat_cortex_plotter(x, y, color_idx, hemispheres, cmaps, **kwargs):

    l_idx = hemispheres == 'L'
    r_idx = hemispheres == 'R'

    # we use these x_mod, y_mod to consistently plot these patches so they are oriented in the same way
    # as the pysurfer cortex plots
    xmax = None
    if l_idx.any() and r_idx.any():
        x_shift = x[l_idx].max() + abs(x[l_idx].min())
        x_shift = x_shift + x_shift/10
        r_cmap = cmaps[r_idx].unique()
        if len(r_cmap) > 1:
            raise Exception('Too many cmaps! Probably faceted wrong...')
        plt.tripcolor(matplotlib.tri.Triangulation(-1*x[r_idx]+x_shift, -1*y[r_idx]), color_idx[r_idx], cmap=r_cmap[0], 
                      shading='gouraud')
        xmax = x_shift + x[r_idx].max() + x_shift/10
        idx = l_idx
        x_mod, y_mod = 1, 1
    elif r_idx.any():
        idx = r_idx
        x_mod, y_mod = -1, -1
    elif l_idx.any():
        idx = l_idx
        x_mod, y_mod = 1, 1
    else:
        raise Exception('Nothing to plot!')

    cmap = cmaps[idx].unique()
    if len(cmap) > 1:
        raise Exception('Too many cmaps! Probably faceted wrong...')
    plt.tripcolor(matplotlib.tri.Triangulation(x_mod*x[idx], y_mod*y[idx]), color_idx[idx], cmap=cmap[0], 
                  shading='gouraud')
    if xmax is None:
        xmax = x[idx].max() + (x[idx].max() + abs(x[idx].min()))/10
    xmin = x[idx].min() - (x[idx].max() + abs(x[idx].min()))/10
    
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim((xmin, xmax))

def make_colorbar(fig, cbar_cmap, nuniq=None, vmin=None, vmax=None, plot_vals=None):
    """
    either plot_vals or all of nuniq, vmin, vmax must be set
    """
    if plot_vals is not None:
        vmin= plot_vals.min()
        vmax = plot_vals.max()
        nuniq = pd.Series(plot_vals).nunique()

    if vmin < 0 and vmax > 0:
        norm = MidpointNormalize(vmin, vmax, 0.)
    else:
        norm = matplotlib.colors.Normalize(vmin, vmax)
    
    cax, _ = matplotlib.colorbar.make_axes(fig.axes, shrink=.5)
    cbar = matplotlib.colorbar.ColorbarBase(ax=cax,cmap=cbar_cmap, norm=norm)

    if nuniq is not None and nuniq < 10:
        ticker = matplotlib.ticker.LinearLocator(numticks=nuniq)
    elif fig.get_figheight() < 4:
        ticker = matplotlib.ticker.MaxNLocator(nbins=5)
    else:
        ticker = None
    
    if ticker is not None:
        cbar.locator = ticker
        cbar.update_ticks()

    return cbar

def plot_flat_cortex(model_df, meshes, data_df=None, plot_val='predicted_responses', hemi='both',
                     facet_col='image_type', facet_row='model', facet_col_vals=None,
                     facet_row_vals=None, set_minmax=True,
                     v123_cmaps= {1: 'Reds', 2: 'Reds', 3: 'Reds'}, **kwargs):
    """Plot the specified property on a flat cortex, averaging across specified categories

    This function plots the property specified by `plot_val` on a flat cortex. `plot_val` can come
    from either `data_df`, in which case it must be the name of one of the columns (e.g.,
    'predicted_responses'), or from `meshes`, in which case it must be the name of a property of
    the mesh (e.g., 'benson14_visual_area'). 

    If `plot_val` is from the mesh, then only one subplot will be made. Otherwise, we will create a
    seaborn FacetGrid, facetting along `facet_col` and `facet_row` (both of which must be columns
    of `data_df`), with one subplot for every combination of the values of `facet_col` and
    `facet_row` (e.g., if each have two values, four subplots are made). Within each subplot, each
    voxel/vertex will have all of its `plot_val` values averaged together. 

    For example, with the following options: `facet_col='image_type'`, `facet_row='model'`,
    `plot_val='predicted_responses'`, let the unique values of `'image_type'` in `data_df` be
    `'original'` and `'V1-metamer'` and the unique values of `'model'` be `'full'` and
    `'dumb_V1'`. Then the first plot will show the predicted response of each voxel in the full
    model, averaged together across all original images (and so on).

    If you don't wish to plot all values of `facet_col` and `facet_row`, use `facet_col_vals` and
    facet_row_vals` to specifiy a subset.

    To change the colormaps used, set `v123_cmaps`.

    Arguments
    ==============
    
    model_df: dataframe summarizing model performance. must contain pRF_vertex_indices and 
              pRF_hemispheres

    meshes: a dictionary of neuropythy projected meshes. Keys should be 'L' and/or 'R'. constructed
            by calling e.g., subject.LH.projection(method='equirectangular')

    data_df: "long form" dataframe, optional. contains the values you want to project on the cortex 
             (e.g., SNR_df, plot_overall_df). If None, it's assumed that `plot_val` is a property 
             mesh and only one subplot will be made (thus, `facet_col` and `facet_row` will be 
             ignored)

    plot_val: str, optional. Default: 'predicted_responses'. what you want to plot. Must either be
              a column in `data_df` or a property of the neuropythy mesh.

    hemi: string, optional. the hemisphere to plot, either 'L'/'LH'/'l'/'lh' or 'R'/'RH'/'r'/'rh'
          or 'both' (default). if 'both', then both will be in the same subplot

    facet_col: str, optional. Default: 'model'. name of the column you want to display as the
               column of the pivot table. Must be a column in `data_df`. Ignored if `plot_val` is a
               property of the mesh
    
    facet_row: str, optional. Default: 'image_type'. name of the column you want to display as the
               row of the pivot table. Must be a column in `data_df`. Ignored if `plot_val` is a 
               property of the mesh

    facet_col_vals: str or list of strs, optional. subset of facet_col that you want to visualize
                    on your columns. Probably values of the model type ('full' and 'dumb_V1'). If
                    None (default), will use all

    facet_row_vals: str or list of strs, optional. subset of facet_row that you want to visualize
                    on your rows. Probably values of the image type (e.g., 'original',
                    'V1-metamer', etc). If None (default), will use all

    set_minmax: boolean, optional. whether all images should have the same vmin and vmax values
                (True) or whether they should be allowed to have separate vmin and vmax values 
                (False)

    v123_cmaps: dictionary or matplotlib colormap, optional. If a dictionary, keys must be 1, 2,
                and 3 and values are the colormaps to use for V1, V2, and V3. by default, all
                three will be matplotlib.cm.Reds. If a colormap, same colormap will be used for all 
                three areas.

    col/row_order can be specified in **kwargs
    """

    if hemi=='both':
        hemi = ['L', 'R']
    else:
        hemi = [_hemi_check(hemi)]
        
    if not isinstance(v123_cmaps, dict):
        v123_cmaps = dict((i, v123_cmaps) for i in range(1,4))
        
    if data_df is not None and facet_col_vals is None:
        facet_col_vals = data_df[facet_col].unique()
    if isinstance(facet_col_vals, basestring):
        facet_col_vals = [facet_col_vals]
        
    if data_df is not None and facet_row_vals is None:
        facet_row_vals = data_df[facet_row].unique()
    if isinstance(facet_row_vals, basestring):
        facet_row_vals = [facet_row_vals]

    vmin, vmax, nuniq, hemi_dfs, pivs = 0, 0, 0, {}, {}
    if data_df is None or plot_val not in data_df.columns:
        try:
            for h in hemi:
                vmin = min(vmin, meshes[h].prop(plot_val).min())
                vmax = max(vmax, meshes[h].prop(plot_val).max())
        except AttributeError:
            raise Exception("Neither your mesh nor your data_df contains plot_val %s!" % plot_val)
        if set_minmax:
            vmin, vmax = float(vmin), float(vmax)
        else:
            vmin, vmax = None, None
        flat_cort_df = []
        for h in hemi:
            map_colors, cmap, nuniq_tmp, cbar_cmap = _flat_cortex_make_cmap(meshes[h], plot_val,
                                                                            vmin=vmin, vmax=vmax, 
                                                                            v123_cmaps= v123_cmaps)
            nuniq = max(nuniq, nuniq_tmp)
            flat_cort_df.append(pd.DataFrame(data={'hemi': h, 'color_map':cmap,
                                                   'x_coords': meshes[h].coordinates[0,:],
                                                   'y_coords': meshes[h].coordinates[1,:]}))
        flat_cort_df = pd.concat(flat_cort_df)
        flat_cort_df = flat_cort_df.reset_index().rename(columns={'index':'color_idx'})
        g = sns.FacetGrid(flat_cort_df, aspect=len(hemi))
        g.map(flat_cortex_plotter, 'x_coords', 'y_coords', 'color_idx', 'hemi', 'color_map')
        if set_minmax:
            make_colorbar(g.fig, cbar_cmap, nuniq, vmin, vmax)
        return g, flat_cort_df
    else:
        # we do this first, because it will be the same for all facet_col, facet_row pairs
        for h in hemi:
            hemi_dfs[h], pivs[h] = _flat_cortex_pivot_table(h, meshes[h], model_df, data_df,
                                                            plot_val, [facet_col, facet_row])
            # that obnoxious second term means that we grab the entries in the corresponding pivot
            # table that we're looking at in this call (the appropriate model, image_type pairs)
            # and determine what their minimum values are (same for vmax)
            vmin = min(vmin, pivs[h][pivs[h].index.isin([i for i in itertools.product(facet_col_vals, facet_row_vals)])].min().min())
            vmax = max(vmax, pivs[h][pivs[h].index.isin([i for i in itertools.product(facet_col_vals, facet_row_vals)])].max().max())
        if set_minmax:
            vmin, vmax = float(vmin), float(vmax)
        else:
            vmin, vmax = None, None

        flat_cort_df = []
        for h, c, r in itertools.product(hemi, facet_col_vals, facet_row_vals):
            summary_vals = _flat_cortex_shape_data(meshes[h], pivs[h], (c, r), hemi_dfs[h])
            map_colors, cmap, nuniq_tmp, cbar_cmap = _flat_cortex_make_cmap(meshes[h], summary_vals,
                                                                            vmin=vmin, vmax=vmax, 
                                                                            v123_cmaps= v123_cmaps)
            nuniq = max(nuniq, nuniq_tmp)
            flat_cort_df.append(pd.DataFrame(data={'hemi': h, 'facet_col': c, 'facet_row': r, 'summary_vals': summary_vals,
                                                   'color_map':cmap, 'x_coords': meshes[h].coordinates[0,:],
                                                   'y_coords': meshes[h].coordinates[1,:]}))
        flat_cort_df = pd.concat(flat_cort_df)
        flat_cort_df = flat_cort_df.reset_index().rename(columns={'index':'color_idx'})
        g = sns.FacetGrid(flat_cort_df, col='facet_col', row='facet_row', aspect=len(hemi), **kwargs)
        g.map(flat_cortex_plotter, 'x_coords', 'y_coords', 'color_idx', 'hemi', 'color_map')
        g.set_titles("{col_name} | {row_name}")
        if set_minmax:
            make_colorbar(g.fig, cbar_cmap, nuniq, vmin, vmax)
        return g, flat_cort_df

def _pysurfer_shape_predictions(hemi, model, image_type, model_df, plot_df, subject):
    """based on export_predicted_response_surface, need to get data in right shape for pysurfer

    hemi: the hemisphere to plot, either 'L'/'LH'/'l'/'lh' or 'R'/'RH'/'r'/'rh'

    model: the model you want to visualize predictions for, must be one of the model values in 
           plot_df (probably 'full' and 'dumb_V1')

    image_type: image type to visualize predictions for, must be one of the image_type values in 
                plot_df (probably, 'original', 'V1-metamer', etc)
    """
    hemi = _hemi_check(hemi)
    hemi_df = model_df[model_df.pRF_hemispheres.isin([hemi])]
    piv = plot_df[plot_df['voxel'].isin(hemi_df['voxel'])]
    piv = piv.pivot_table(values='predicted_responses', columns='voxel', index=['model', 'image_type'], aggfunc=np.mean)
    if hemi=='L':
        preds = np.full((subject.LH.vertex_count), 0., dtype=np.float32)
    elif hemi=='R':
        preds = np.full((subject.RH.vertex_count), 0., dtype=np.float32)
    pidcs = hemi_df['pRF_vertex_indices'].values
    preds[pidcs] = piv.loc[model, image_type].values
    return preds, pidcs

def pysurfer_plot_predictions(subject_id, hemi, model, image_type, model_df, plot_df,
                              mode='data', save_path='brain.png', cmap='Reds', subjects_dir=None,
                              **kwargs):
    """plot average predictions on pysurfer brain and save to image
    
    this is a little limited, since I don't think it's as useful a visualization. but it will
    average across all predicted responses for the specified model and image (similar to
    `plot_flat_cortex`), allowing you to visualize how the predicted responses are distributed
    across the visual cortex.
    
    this takes a bit of time to run.

    This requires the optional library pysurfer to run.
    
    KNOWN BUG: this function may not properly save the image by itself. If this happens (the image
    will be saved but will not show the overlaid brain), call `brain.save_image(path)` with the
    returned Brain object.
    
    hemi: the hemisphere to plot, either 'L'/'LH'/'l'/'lh' or 'R'/'RH'/'r'/'rh'
    
    model: the model you want to visualize predictions for, must be one of the model values in 
           plot_df (probably 'full', 'dumb_V1', etc)

    image_type: image type to visualize predictions for, must be one of the image_type values in 
                plot_df (probably, 'original', 'V1-metamer', etc)
                
    mode: {'data', 'overlay'}, whether to use surfer.Brain.add_data or surfer.Brain.add_overlay to
          add the data to the Brain object. 'data' allows a colormap to be specified, while 'overlay'
          does not, but 'overlay' will only show relevant values on the brain, plotting all vertices
          without values as transparent, while 'data' will show them as the minimum value on the
          colormap, which is distracting.

    returns `brain`, the pysurfer Brain with data projected on top. It is recommended that you
    close the object (call `brain.close()`) when you have finished with it
    """
    hemi = _hemi_check(hemi, 'surfer')
    view_dict = {'lh': {'azimuth': 290., 'elevation': 120., 'roll': 0.},
                 'rh': {'azimuth': 260., 'elevation': 120., 'roll': 0.}}
    subject = neuropythy.freesurfer_subject(subject_id)
    preds, pidcs = _pysurfer_shape_predictions(hemi, model, image_type, model_df, plot_df, subject)
    brain = Brain(subject_id, hemi, 'inflated', views=view_dict[hemi], subjects_dir=subjects_dir)
    vmin = kwargs.pop('min', .0001)
    if mode=='data':
        brain.add_data(preds, min=vmin, max=preds.max(), colormap=cmap, vertices=pidcs, **kwargs)
    elif mode=='overlay':
        brain.add_overlay(preds, min=vmin, max=preds.max(), **kwargs)
    else:
        raise Exception("Don't know how to add data to Brain with mode %s!" % mode)
    brain.save_image(save_path)
    return brain
