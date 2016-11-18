# freq_pref_check.py
#
# This script checks whether the pRFs the models are working the way they should be.
#
# by William F. Broderick

import argparse
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
    new_stimuli['horiz_sweep'].extend(new_stimuli['vert_sweep'])
    # extend is always in place, so this will work, but it just will be in horiz_sweep, which is a
    # little janky
    sp.io.savemat(save_template_string.format("sweep"),
                  {'images': np.asarray(new_stimuli['horiz_sweep'])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Create the stimuli necessary to test the pRF "
                                                  "location and size."))
    parser.add_argument("stimuli_path", help=("string. path to the stimuli to modify. Should be "
                                              "the stimuli.mat from Kendrick's website"))
    parser.add_argument("save_template_string", help=("string containing one {}. The string that "
                                                      "we'll save our stimuli at, using .format"))
    args = parser.parse_args()
    create_stimuli(args.stimuli_path, args.save_template_string)
