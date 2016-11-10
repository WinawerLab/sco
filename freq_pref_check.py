# freq_pref_check.py
#
# This script checks whether the frequency preferences of the model are working the way they should
# be.
#
# by William F. Broderick

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sco.model_comparison import (create_model_dataframe, compare_with_Kay2013, _create_plot_df)

def generate_stimuli(img_folder, freqs):
    if img_folder[-1] != '/':
        img_folder += '/'
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)
    img_name = "freq_{}_grating.png"
    x = np.asarray(range(1000))/1000.0
    x = np.meshgrid(x,x)[0]
    for fr in freqs:
        plt.imsave(img_folder+img_name.format(fr), np.sin(fr*2*np.pi*x))


def main(model_df_path="./sco_freq_prefs.csv", subject='test-sub', subject_dir=None,
         img_folder="~/Desktop/freq_pref_imgs"):
    img_folder = os.path.expanduser(img_folder)
    freqs = np.asarray(range(5,65,5))
    generate_stimuli(img_folder, freqs+1)
    results, stimulus_model_names = compare_with_Kay2013(img_folder, range(len(freqs)), range(3), subject,
                                                         subject_dir)
    model_df = create_model_dataframe(results, stimulus_model_names, model_df_path)

    plot_df = _create_plot_df(freqs, model_df)
    plot_df = plot_df[plot_df.language == 'python']
    plot_df['frequency'] = plot_df['image'].apply(lambda x: int(re.search(r'freq_([0-9]+)', x).groups()[0]))

    g = sns.factorplot(data=plot_df, x='frequency', y='predicted_responses')
    
    return model_df, results
