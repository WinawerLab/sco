# metamers.py
#
# code specifically for evaluating SCO model performance on "texture metamers", as discussed in
# Freeman2013 and Portilla2000. See Metamers.ipynb for more information.
#
# References:
# - Freeman, J., Ziemba, C. M., Heeger, D. J., Simoncelli, E. P., & Movshon, J. A. (2013). A
#   functional and perceptual signature of the second visual area in primates. Nature Neuroscience.
# - Portilla, J., & Simoncelli, E. P. (2000). A parametric texture model based on joint statistics of
#   complex wavelet coefficients. International journal of computer vision.
# 
# by William F. Broderick

import re
import os
import neuropythy
import pickle
import pandas as pd
from .core import save_results_dict, create_SNR_df
from ..model_comparison import create_model_dataframe, _create_plot_df
from .. import calc_surface_sco

def search_for_mets(x, re_exp=r'(V[12]Met(Scaled)?).*'):
    """find the metamer type in the filename
    """
    tmp = re.search(re_exp, x)
    if tmp is None:
        return "original"
    else:
        return tmp.groups()[0].replace('MetScaled', 'SclMet').replace('Met', '-metamer')

def search_for_noise_seed(x, re_exp=r'im[0-9]+-smp1-([0-9]+).*png'):
    """find the noise seed in the filename
    """
    tmp = re.search(re_exp, x)
    if tmp is None:
        return None
    else:
        return tmp.groups()[0]

def create_met_df(df, col_name='image', type_regex=r'(V[12]Met(Scaled)?).*',
                  seed_regex=r'im[0-9]+.*-([0-9]+).*png', name_regex=r'(im[0-9]+).*png'):
    """for each image, determine its type, name, and seed

    These are all based on regex matching with the image name, so the assumption is that images
    have been named in a straightforward manner. If you used createMetamers.m / Makefile to create
    your stimuli, then the default values should work. Otherwise, you'll have to set them by
    hand.
    """
    if 'language' in df.columns:
        df = df[df.language=='python']
    df['image_type'] = df[col_name].apply(search_for_mets, args=(type_regex,))
    df['image_name'] = df[col_name].apply(lambda x: re.search(name_regex, x).groups()[0])
    df['image_seed'] = df[col_name].apply(search_for_noise_seed, args=(seed_regex,))
    return df

def create_image_df(results):
    """given the results dict, return all the images, annotated in a dataframe (for easy facetting)
    
    currently, this only works for metamers, because it tries to find the image_seed, image_type,
    and image_name, which all are defined in a metamer-specific way. however, in the future it may
    be useful to make this more general; I need to see another usecase before how to do so becomes
    clear.

    note that this can take a while to run.
    """
    def make_df(img, norm_img, idx, filename):
        t = pd.DataFrame(img).unstack()
        t2 = pd.DataFrame(norm_img).unstack()
        tmp = pd.concat([t, t2], axis=1).reset_index()
        tmp = tmp.rename(columns={0: 'value', 1: 'norm_value'})
        tmp['image_idx'] = idx
        tmp['filename'] = filename
        tmp['image_seed'] = search_for_noise_seed(filename)
        tmp['image_type'] = search_for_mets(filename)
        tmp['image_name'] = re.search(r'(im[0-9]+)-smp1.*png', filename).groups()[0]
        return tmp
    
    filenames = [os.path.split(name)[1] for name in results['stimulus_image_filenames']]
    image_df = pd.concat([make_df(img, norm_img, i, f) for i, (img, norm_img, f) in enumerate(zip(results['stimulus_images'], results['normalized_stimulus_images'], filenames))])
    image_df['image_seed'] = image_df['image_seed'].replace({None: '1'})
    return image_df

def main(images, output_dir, model_name='full', model_steps=['results', 'model_df', 'SNR_df'],
         **kwargs):
    """Run the SCO model on metamers

    NOTE that you will likely need to set type_regex, seed_regex, and name_regex so they correctly
    parse the names of your files. The defaults will work if your original files are named
    'im##.png', where ## can be any number of digits, and you used createMetamers.m to create your
    metamers. This results in metamers being called V1Met-im##-#.png, V2Met-im##-#.png, and
    V2MetScaled-img##-#.png, where the final -# specifies the random seed. If your filenames are
    formatted differently, you'll need to set these regex expression by hand (can be done
    SCO_KWARGS or if main block in cluster_submit.py).

    Three outputs are created as part of this call: results, model_df, and SNR_df. Each requires
    the previous one, but they can be created in separate calls (loading in the previous ones). To
    specify the ones you wish to run in this call, use model_steps
    """
    if 'subject_path' in kwargs:
        neuropythy.freesurfer.add_subject_path(os.path.expanduser(kwargs.pop('subject_path')))
    if isinstance(model_steps, basestring):
        model_steps = [model_steps]
    subject = kwargs.get('subject', 'test-sub')
    max_eccentricity = kwargs.get('max_eccentricity', 7.5)
    bootstrap_num = kwargs.get('bootstrap_num', 100)
    type_regex = kwargs.get('type_regex', r'(V[12]Met(Scaled)?).*')
    seed_regex = kwargs.get('seed_regex', r'im[0-9]+.*-([0-9]+).*png')
    name_regex = kwargs.get('name_regex', r'(im[0-9]+).*png')
    print("Running %s" % model_name)
    if 'results' in model_steps:
        results = calc_surface_sco(subject=subject, stimulus_image_filenames=images,
                                   max_eccentricity=max_eccentricity, **kwargs)
        save_results_dict(results, '%s/results_%s.pkl' % (output_dir, model_name))
    else:
        with open('%s/results_%s.pkl' % (output_dir, model_name)) as f:
            results = pickle.load(f)
    if 'model_df' in model_steps:
        images = [os.path.split(i)[1] for i in images]
        model_df = create_model_dataframe(results, images, '%s/model_df_%s.csv' % (output_dir, model_name))
    else:
        model_df = pd.read_csv('%s/model_df_%s.csv' % (output_dir, model_name))
    # extra_cols is a list of strings corresponding to columns in model_df that you would also like
    # to add to the plot_df
    plot_df = _create_plot_df(model_df, extra_cols=['Kay2013_output_nonlinearity', 'Kay2013_SOC_constant'])
    plot_df = create_met_df(plot_df, type_regex=type_regex, seed_regex=seed_regex,
                            name_regex=name_regex)
    if 'SNR_df' in model_steps:
        SNR_df = create_SNR_df(plot_df, bootstrap_num=bootstrap_num,
                               file_name='%s/SNR_df_%s.csv' % (output_dir, model_name),
                               extra_cols=['Kay2013_output_nonlinearity', 'Kay2013_SOC_constant'])
    else:
        SNR_df = pd.read_csv('%s/SNR_df_%s.csv' % (output_dir, model_name))
    return results, model_df, plot_df, SNR_df
