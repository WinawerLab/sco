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
    model_df = create_model_dataframe(results, stimulus_model_names, model_df_path)
    return model_df, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
