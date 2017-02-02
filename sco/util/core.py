####################################################################################################
# sco/util/core.py
# Utilities useful in work related to the sco predictions.
# By William F Broderick

import itertools
import pickle
import neuropythy
import pandas as pd

def _bootstrap_voxel_df(df, extra_unique_cols=['image_name']):
    """bootstrap the rows of `df`
    """
    try:
        df = df.groupby(['voxel']+extra_unique_cols).mean().reset_index()
    except TypeError:
        df = df.groupby(['voxel']+[extra_unique_cols]).mean().reset_index()
    sample_num = df[df.voxel == 0].shape[0]
    boot_idx = df[df.voxel == 0].reset_index().sample(sample_num, replace=True).index
    gb = df.groupby('voxel')
    resampled_df = gb.apply(lambda g: g.iloc[boot_idx])
    return resampled_df

def _create_bootstrap_df(bootstrap_num, plot_df, bootstrap_val, diff_col, extra_unique_cols):
    """bootstrap `bootstrap_val` `bootstrap_num` times, getting the difference between each pair of
    `diff_col` values
    """
    bootstrap_df = []
    df_dicts = dict((k, plot_df[plot_df[diff_col] == k]) for k in plot_df[diff_col].unique())
    for i in range(bootstrap_num):
        bootstrapped_dfs_dict = dict((k, _bootstrap_voxel_df(df_dicts[k], extra_unique_cols).set_index(['voxel', 'image_name'])) for k in df_dicts.keys())

        tmp_dfs = []
        for name1, name2 in itertools.combinations(bootstrapped_dfs_dict.keys(), 2):
            tmp_df = bootstrapped_dfs_dict[name1][[bootstrap_val]].mean(level=(0)) - bootstrapped_dfs_dict[name2][[bootstrap_val]].mean(level=(0))
            tmp_df = tmp_df.rename(columns={bootstrap_val: '%s - %s' % (name1, name2)})
            tmp_dfs.append(tmp_df)

        tmp_dfs = pd.concat(tmp_dfs, 1).reset_index()
        tmp_dfs['bootstrap_num'] = i
        bootstrap_df.append(tmp_dfs)

    return pd.concat(bootstrap_df).rename(columns=lambda x: x.replace('-metamer', ''))

def create_SNR_df(plot_df, bootstrap_val='predicted_responses', diff_col='image_type',
                  extra_unique_cols=['image_name'], bootstrap_num=100, file_name='SNR_df.csv',
                  extra_cols=[]):
    """Create a dataframe with bootstrapped signal-to-noise ratio values.

    the signal-to-noise ratio is a measure of the difference in a given value between
    conditions. `plot_df` will be split into the unique values of `diff_col`, then averaged so
    every combination of voxels and values of `extra_unique_cols` have one row, then bootstrapped,
    taking the difference in the averages of `bootstrap_val` between each pair of `diff_col`
    values. This will be done `bootstrap_num` times, then the average over the standard deviation
    of these `bootstrap_num` values will be recorded as the SNR. Each voxel will therefore have n
    choose 2 values, where `n = plot_df[diff_col].nunique()`, one for each pair of `diff_col`
    values.

    This will also save the returned `SNR_df` at `file_name`, since this takes a while to run.
    
    extra_cols is a list of strings corresponding to columns in plot_df that you would also like
    to add to the SNR_df
"""
    bootstrap_df = _create_bootstrap_df(bootstrap_num, plot_df, bootstrap_val, diff_col, extra_unique_cols)

    gb = bootstrap_df.groupby('voxel')

    SNR_df = plot_df.drop_duplicates(subset='voxel')[['v123_label']+extra_cols]
    SNR_df.index.name = 'voxel'
    for col in [col for col in bootstrap_df.columns if '-' in col]:
        SNR_df[col] = gb.apply(lambda g: g[[col]].mean() / g[[col]].std())

    SNR_df = pd.melt(SNR_df.reset_index(), id_vars=['voxel', 'v123_label'])

    SNR_df.to_csv(file_name)

    return SNR_df

def save_results_dict(results, file_name='results.pkl'):
    """we want to save the results dictionary but we can't pickle functions. We need to do this

    because our results dict contains both functions and arrays of functions. So we go through and
    save those entries in our dictionary that are not callable.
    """
    save_results = dict()
    for k, v in results.iteritems():
        # we can't pickle the free surfer subject
        if isinstance(v, neuropythy.freesurfer.subject.Subject):
            continue
        # we can't pickle functions or lambdas
        if _check_uncallable(v):
            save_results[k] = v
    with open(file_name, 'w') as f:
        pickle.dump(save_results, f)

def _check_uncallable(x):
    """if this is an iterable, we want to call this again (we don't know the structure of this
    iterable, so x[0] might be another iterable).
    """
    if hasattr(x, '__iter__'):
        # we can have an array that just contains None, in which case v[0] will throw an
        # exception
        try:
            return _check_uncallable(x[0])
        except IndexError:
            return False
        # if this is a dictionary, it may return a keyerror, in which case we want to check the
        # first value.
        except KeyError:
            return _check_uncallable(x.values()[0])
    else:
        if not hasattr(x, '__call__'):
            return True
        else:
            return False
