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

def _search_for_label(x, regex, none_val):
    """search filename x for label using regex, returning none_val if nothing is found
    """
    tmp = re.search(regex, x)
    if tmp is None:
        return none_val
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
    df['image_type'] = df[col_name].apply(_search_for_label, args=(type_regex, 'original'))
    df['image_type'] = df['image_type'].apply(lambda x: x.replace('MetScaled', 'SclMet').replace('Met', '-metamer'))
    df['image_name'] = df[col_name].apply(_search_for_label, args=(name_regex, None))
    df['image_seed'] = df[col_name].apply(_search_for_label, args=(seed_regex, None))
    return df

def create_image_struct(results, regex_dict={'image_type': (r'(V[12]Met(Scaled)?).*', 'original'),
                                             'image_seed': (r'im[0-9]+.*-([0-9]+).*png', '1'), 
                                             'image_name': (r'(im[0-9]+).*png', None)},
                        replace_dict={'image_type': [('MetScaled', 'SclMet'), ('Met', '-metamer')]}):
    """given the results dict, return all the images with labels (for easy facetting)
    
    This returns a dictionary with the following keys:

    - stimulus_images: the images array from the results dict

    - normalized_stimulus_images: the normalized images array from the results dict

    - labels: a pandas DataFrame containing labels of the images. The main two columns are filenames
              and image_index. filenames is the filename of the image, without any folders and 
              image_index is the appropriate index into stimulus_images and normalized_stimulus_images 
              to find the corresponding image. It will also contain a column for each entry in `regex_dict`,
              storing the result of searching filenames for the value in a column labeled with the key. 

    this plays well with plot_images, allowing one to easily find and plot images in an easy-to-parse
    way

    each entry in regex_dict should have two values: the first a regex to search filenames with, the second
    the value that should be inserted in case the regex comes up empty
    
    replace_dict allows one to optionally replace values from one column. After a column is created, if its
    name shows up as a key in replace_dict, we will call label_df[key].apply(lambda x: x.replace(*replace_dict[key])) 
    (so the value should be a tuple or a list of tuples, in which case each is called one in turn)
    """
    filenames = [os.path.split(name)[1] for name in results['stimulus_image_filenames']]
    label_df = pd.DataFrame({'filenames': filenames})
    for col, (regex, none_val) in regex_dict.iteritems():
        label_df[col] = label_df['filenames'].apply(_search_for_label, args=(regex, none_val))
        if col in replace_dict:
            if hasattr(replace_dict[col][0], '__iter__'):
                for v in replace_dict[col]:
                    label_df[col] = label_df[col].apply(lambda x: x.replace(*v))
            else:
                label_df[col] = label_df[col].apply(lambda x: x.replace(*replace_dict[col]))
    label_df = label_df.reset_index().rename(columns={'index': 'image_idx'})
    images = {'labels': label_df, 'stimulus_images': results['stimulus_images']}
    if 'normalized_stimulus_images' in results.keys():
        images['normalized_stimulus_images'] = results['normalized_stimulus_images']
    return images

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
    bootstrap_num = kwargs.get('bootstrap_num', 1000)
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
