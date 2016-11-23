####################################################################################################
# model_comparison_script.py
#
# this allows us to run the different model_comparison functions as scripts, so we can integrate
# them into a larger workflow. I can't include this as part of the compare_with_Kay2013.py (for
# example), because then relative imports don't work in that file, which we need.
#
# for now, this just wraps around compare_with_Kay2013. could eventually add several functions,
# adding flags / args to differentiate between them.
#
# By William F. Broderick

import argparse
import pickle
import os
import neuropythy
import numpy as np
from sco.model_comparison import (create_model_dataframe, compare_with_Kay2013)


def main(image_base_path, stimuli_idx, voxel_idx, subject, model_df_path="./soc_model_params.csv",
         subject_dir=None):
    """wrapper for compare_with_Kay2013

    this function simply acts as a wrapper around compare_with_Kay2013, allowing one to call it on
    the command-line. See that function (within sco/model_comparison/compare_with_Kay2013.py) for
    more details.
    """
    if voxel_idx is not None:
        voxel_idx = np.asarray(voxel_idx)
    results, stimulus_model_names = compare_with_Kay2013(image_base_path, np.asarray(stimuli_idx),
                                                         voxel_idx, subject,
                                                         subject_dir)
    # we want to save the results dictionary but we can't pickle functions. We need to do this
    # because our results dict contains both functions and arrays of functions.
    save_results = dict()
    for k, v in results.iteritems():
        # we can't pickle the free surfer subject
        if isinstance(v, neuropythy.freesurfer.subject.Subject):
            continue
        # we can't pickle functions or lambdas
        if _check_uncallable(v):
            save_results[k] = v
    with open(os.path.splitext(model_df_path)[0] + "_results_dict.pkl", 'w') as f:
        pickle.dump(save_results, f)

    model_df = create_model_dataframe(results, stimulus_model_names, model_df_path)
    return model_df, results

def _check_uncallable(x):
    # if this is an iterable, we want to call this again (we don't know the structure of this
    # iterable, so x[0] might be another iterable).
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Create the dataframe with predicted responses for"
                                                  " the given stimuli, voxels, and subject, using "
                                                  "the options from Kay2013, for comparison with "
                                                  "the matlab version of model."))
    parser.add_argument(
        "image_base_path",
        help=("string. This is assumed to either be a directory, in which case the model will be "
              "be run on every image in the directory, or a .mat file containing the image stimuli"
              ", in which case they will be loaded in and used."))
    parser.add_argument("subject", help=("string. The specific subject to run on."))
    parser.add_argument("model_df_path", help=("string. Absolute path to save the model dataframe"
                                               " at."))
    parser.add_argument("stimuli_idx", nargs='+', type=int,
                        help="list of ints. Which indices in the stimuli to run.")
    parser.add_argument("-v", "--voxel_idx", nargs='+', type=int, default=None,
                        help="list of ints. Which voxels to run. If not specified, will run all")
    parser.add_argument("-s", "--subject_dir", help=("string (optional). If specified, will add to"
                                                     "neuropythy.freesurfer's subject paths"),
                        default=None)
    args = parser.parse_args()
    main(args.image_base_path, args.stimuli_idx, args.voxel_idx, args.subject,
         args.model_df_path, args.subject_dir)
