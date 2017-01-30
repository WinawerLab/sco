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
from . import save_results_dict, create_SNR_df
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
                  seed_regex=r'im[0-9]+-smp1-([0-9]+).*png', name_regex=r'(im[0-9]+)-smp1.*png'):
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


def main(images, output_dir, model_name='full', **kwargs):
    """Run the SCO model on metamers
    """
    if 'subject_path' in kwargs:
        neuropythy.freesurfer.add_subject_path(os.path.expanduser(kwargs.pop('subject_path')))
    subject = kwargs.get('subject', 'test-sub')
    max_eccentricity = kwargs.get('max_eccentricity', 7.5)
    bootstrap_num = kwargs.get('bootstrap_num', 100)
    sample_num = kwargs.get('sample_num', 50)
    results = calc_surface_sco(subject=subject, stimulus_image_filenames=images,
                               max_eccentricity=max_eccentricity, **kwargs)
    save_results_dict(results, '%s/results_%s.pkl' % (output_dir, model_name))

    images = [os.path.split(i)[1] for i in images]
    model_df = create_model_dataframe(results, images, '%s/model_df_%s.csv' % (output_dir, model_name))
    plot_df = _create_plot_df(model_df)
    plot_df = create_met_df(plot_df)
    SNR_df = create_SNR_df(plot_df, bootstrap_num=bootstrap_num, sample_num=sample_num,
                           file_name='%s/SNR_df_%s.csv' % (output_dir, model_name))
    return results, model_df, plot_df, SNR_df
