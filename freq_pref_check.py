# freq_pref_check.py
#
# This script checks whether the frequency preferences of the model are working the way they should
# be.
#
# by William F. Broderick

import re
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sco.model_comparison import (create_model_dataframe, compare_with_Kay2013, _create_plot_df)

def crossfade_img(img, cpd=0.25, res=12):
    """from Noah, to get those horizontal fading bars.

    res should be the pixels per degree of your stimulus image (stimulus_pixels_per_degree) and
    cpd should be the target cpd of your crossfaded bars (this should generally be quite low).
    
    the default cpd works with the default res; I've found cpd=.1 to work with res=1000./15
    """
    import numpy.matlib as npm
    w = img.shape[0]
    col = 0.5*(1 + np.cos(np.asarray([range(w)]).T * cpd/res*2*np.pi))
    mat = np.matlib.repmat(col, 1, w)
    return 0.5 + (img - 0.5)*mat

def generate_stimuli(img_folder, freqs, img_res, stimd2p):
    if img_folder[-1] != '/':
        img_folder += '/'
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)
    img_name = "freq_{:02d}_grating.png"
    x = np.asarray(range(img_res))/float(img_res)
    x = np.meshgrid(x,x)[0]
    for fr in freqs:
        # because we have the second order contrast calculation in our model, if we just pass a
        # uniform grating, the response to most of the image will be 0 and our response will be
        # driven by edge effects. We want to avoid this, so we "zero-out" (actually, use the middle
        # value) several stripes in the image so we have contrast in several locations.
        signal = np.sin(fr*2*np.pi*x)
        # make this based on a low frequency sinusoid so the effect is a little less harsh.
        signal = crossfade_img(signal, .1, stimd2p)
        plt.imsave(img_folder+img_name.format(fr), signal)


def make_plot(model_df, img_size_in_pixels, img_size_in_degrees):
    plot_df = _create_plot_df(model_df)
    plot_df = plot_df[plot_df.language == 'python']
    # in the naive case, we assume none of the image was cut out of the aperture, and so the number
    # of cycles per image is given by the frequency of the generating sine wave.
    plot_df['cycles_per_full_image'] = plot_df['image'].apply(lambda x: int(re.search(r'freq_([0-9]+)', x).groups()[0]))
    # we try to grab the pixels per degree, if it's there.
    # try:
    plot_df['pix_per_deg'] = plot_df['image'].apply(lambda x: float(re.search(r'pix_per_deg_([\.0-9]+)', x).groups()[0]))
    plot_df['cycles_per_norm_image'] = plot_df.apply(lambda x: x['cycles_per_full_image']*(x['pix_per_deg']/(img_size_in_pixels/img_size_in_degrees)), axis=1)
    # except AttributeError:
    #     pass
    plot_df['cycles_per_degree'] = plot_df['cycles_per_norm_image'].astype(float)/img_size_in_degrees
    # want to make this look good
    plot_df['cycles_per_degree'] = plot_df['cycles_per_degree'].map(lambda x: "{:.02f}".format(x))
    # try:
    plot_df['preferred_frequency'] = plot_df['image'].apply(lambda x: float(re.search(r'freq_pref_([\.0-9]+)', x).groups()[0]))
    g = sns.factorplot(data=plot_df, x='preferred_frequency', y='predicted_responses', size=8,
                       hue='cycles_per_degree')
    # except AttributeError:
    #     g = sns.factorplot(data=plot_df, x='cycles_per_degree', y='predicted_responses', size=8)
    

    return g, plot_df
        

def check_pref_across_frequencies(img_folder, output_img_path, model_df_path, preferred_freq, stimuli_idx,
                                  stimulus_pixels_per_degree, max_eccentricity=7.5, img_size=1000,
                                  **kwargs):
    """this runs the model with one preferred frequency across images with differing cycles per degree
    """
    def freq_pref(e, s, l):
        # This takes in the eccentricity, size, and area, but we don't use any of them, since we
        # just want to use 1 cpd (for testing) and ignore everything else. And this must be floats.
        return {float(preferred_freq): 1.0}
    results, stimulus_model_names = compare_with_Kay2013(
        img_folder, stimuli_idx, range(3), pRF_frequency_preference_function=freq_pref,
        max_eccentricity=max_eccentricity, **kwargs)
    stimulus_model_names = ["freq_pref_{:02f}_pix_per_deg_{:02f}_".format(stimulus_pixels_per_degree,
                                                                          preferred_freq) + name for name in stimulus_model_names]
    model_df = create_model_dataframe(results, stimulus_model_names, model_df_path)

    # g, plot_df = make_plot(model_df, img_size, 2*max_eccentricity)
    # g.savefig(output_img_path)
    
    return model_df, results#, plot_df, g


def check_pref_across_degrees_per_pixel(img_folder, output_img_path, model_df_path, preferred_freq,
                                        stimuli_idx, stimulus_pixels_per_degree,
                                        max_eccentricity=7.5, img_size=1000, **kwargs):
    """this runs the model with one preferred frequency on one image with differing pixels per degree
    """
    # we want to "zoom in", so we can decrease the stimulus pixels per degree, but not increase.
    stimulus_pixels_per_degree = np.asarray([(i+1)*stimulus_pixels_per_degree/2 for i in range(2)])
    # stimulus_pixels_per_degree = np.asarray([(i+1)*stimulus_pixels_per_degree/10 for i in range(10)])
    model_df = []
    results = {}
    for d2p in stimulus_pixels_per_degree:
        model_df_tmp, results[d2p] = check_pref_across_frequencies(
            img_folder, output_img_path, model_df_path, preferred_freq, stimuli_idx,
            img_size=img_size, max_eccentricity=max_eccentricity, stimulus_pixels_per_degree=d2p,
            **kwargs)
        # results[d2p], stimulus_model_names = compare_with_Kay2013(
        #     img_folder, stimuli_idx, range(3), pRF_frequency_preference_function=freq_pref,
        #     max_eccentricity=max_eccentricity, stimulus_pixels_per_degree=d2p, **kwargs)
        # this will only have one value in it
        # assert len(stimulus_model_names)==1, "This should only have one value here, something went wrong"
        # stimulus_model_names[0] = "pix_per_deg_{:.02f}_".format(d2p) + stimulus_model_names[0]
        # model_df.append(create_model_dataframe(results[d2p], stimulus_model_names, model_df_path))
        model_df.append(model_df_tmp)
    # This annoying bit of code combines the dataframes found here and drops all duplicate columns
    # (stackoverflow.com/questions/16938441/how-to-remove-duplicate-columns-from-a-dataframe-using-python-pandas)
    model_df = pd.concat(model_df, axis=1).T.groupby(level=0).first().T
    model_df.to_csv(model_df_path, index_label='voxel')

    # g, plot_df = make_plot(model_df, img_size, 2*max_eccentricity)
    # g.savefig(output_img_path)
    
    return model_df, results#, plot_df, g


def check_response_across_prefs(img_folder, output_img_path, model_df_path,
                                stimuli_idx, max_eccentricity=7.5, img_size=1000, **kwargs):
    """this runs the model with several preferred frequencies on one image with one pixels per degree
    """
    # def freq_pref(e, s, l, freq):
    #     # This takes in the eccentricity, size, and area, but we don't use any of them, since we
    #     # just want to use 1 cpd (for testing) and ignore everything else. And this must be floats.
    #     return {float(freq): 1.0}
    # def freq_pref_wrapper(freq):
    #     return lambda e,s,l: freq_pref(e,s,l,freq)
    model_df = []
    results = {}
    # for freq_iter in range(10):
    for freq_iter in range(2):
        freq = (freq_iter+1)/5.
        # freq_pref_func = freq_pref_wrapper(freq)
        model_df_tmp, results[freq] = check_pref_across_degrees_per_pixel(
            img_folder, output_img_path, model_df_path, freq, stimuli_idx,
            max_eccentricity=max_eccentricity, img_size=img_size, **kwargs)
        model_df.append(model_df_tmp)
        # results[freq], stimulus_model_names = compare_with_Kay2013(
        #     img_folder, stimuli_idx, range(3), pRF_frequency_preference_function=freq_pref_func,
        #     max_eccentricity=max_eccentricity, **kwargs)
        # this will only have one value in it
        # assert len(stimulus_model_names)==1, "This should only have one value here, something went wrong"
        # stimulus_model_names[0] = "freq_pref_{:.02f}_".format(freq) + stimulus_model_names[0]
        # model_df.append(create_model_dataframe(results[freq], stimulus_model_names, model_df_path))
    # This annoying bit of code combines the dataframes found here and drops all duplicate columns
    # (stackoverflow.com/questions/16938441/how-to-remove-duplicate-columns-from-a-dataframe-using-python-pandas)
    model_df = pd.concat(model_df, axis=1).T.groupby(level=0).first().T
    model_df.to_csv(model_df_path, index_label='voxel')

    # g, plot_df = make_plot(model_df, img_size, 2*max_eccentricity)
    # g.savefig(output_img_path)
    
    return model_df, results#, plot_df, g
    

def main(model_df_path="./sco_freq_prefs.csv", subject='test-sub', subject_dir=None,
         img_folder="~/Desktop/freq_pref_imgs", output_img_path="./sco_freq_prefs.svg",
         stimulus_pixels_per_degree=None, normalized_pixels_per_degree=12):
    img_folder = os.path.expanduser(img_folder)
    if os.path.isdir(img_folder):
        shutil.rmtree(img_folder)
    img_res = 1000
    # 2*max_eccentricity is the size of the image in degrees
    max_eccentricity = 7.5
    freqs = np.asarray(range(30))
    if stimulus_pixels_per_degree is None:
        stimulus_pixels_per_degree = float(img_res) / (2*max_eccentricity)
    generate_stimuli(img_folder, freqs+1, img_res, stimulus_pixels_per_degree)
    model_kwargs = dict(subject=subject, subject_dir=subject_dir,
                  stimulus_pixels_per_degree=stimulus_pixels_per_degree,
                  normalized_pixels_per_degree=normalized_pixels_per_degree,
                  # in order to avoid having edge effects be very large, we set
                  # stimulus_aperture_edge_width to 20 to get some smoothness between the image and
                  # the surrounding area.
                  stimulus_aperture_edge_width=20, max_eccentricity=max_eccentricity)
    # model_df_paf, results_paf, plot_df_paf, g_paf = check_pref_across_frequencies(
    #     img_folder, freqs, os.path.splitext(output_img_path)[0]+"_pref_across_freqs.svg",
    #     os.path.splitext(model_df_path)[0]+"_pref_across_freqs.csv", img_size=img_res, **model_kwargs)
    # model_df, results, plot_df, g = check_pref_across_degrees_per_pixel(
    #     img_folder, os.path.splitext(output_img_path)[0]+"_pref_across_d2p.svg",
    #     os.path.splitext(model_df_path)[0]+"_pref_across_d2p.csv", np.array([29]), img_size=img_res,
    #     **model_kwargs)
    # model_df, results, plot_df, g = check_response_across_prefs(
    #     img_folder, os.path.splitext(output_img_path)[0]+"_resp_across_prefs.svg",
    #     os.path.splitext(model_df_path)[0]+"_resp_across_prefs.csv", np.array([14]), img_size=img_res,
    #     **model_kwargs)
    model_df, results = check_response_across_prefs(
        img_folder, output_img_path, model_df_path, np.array(range(2)), img_size=img_res,
        **model_kwargs)

    return model_df, results#, plot_df, g
