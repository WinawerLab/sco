# freq_pref_check.py
#
# This script checks whether the pRFs the models are working the way they should be.
#
# by William F. Broderick

import scipy as sp
import numpy as np
import sco


def create_stimuli(stimuli_path, save_template_string):
    stimuli = sco.model_comparison.core._load_pkl_or_mat(stimuli_path, 'images')[0, :]
    new_stimuli = {'horiz_sweep': [stimuli[69]], 'vert_sweep': [stimuli[100]]}
    for i, j in zip(range(69, 84), range(100, 115)):
        # rescaling the stimuli to -.5, .5 makes this easier
        tmp = (stimuli[i+1]/255. - .5) - (stimuli[i]/255. - .5)
        new_stimuli['horiz_sweep'].append((tmp+.5)*255.)
        tmp = (stimuli[j+1]/255. - .5) - (stimuli[j]/255. - .5)
        new_stimuli['vert_sweep'].append((tmp+.5)*255.)
    new_stimuli['horiz_sweep'].append(stimuli[99])
    new_stimuli['vert_sweep'].append(stimuli[130])
    for name in ['vert_sweep', 'horiz_sweep']:
        sp.io.savemat(save_template_string.format(name),
                      {'images': np.asarray(new_stimuli[name])})


