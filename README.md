# sco ##############################################################################################
The Standard Cortical Observer Python library.

## Author ##########################################################################################
Noah C. Benson &lt;<nben@nyu.edu>&gt;

## Dependencies ####################################################################################

The neuropythy library depends on two other libraries, all freely available:
 * [numpy](http://numpy.scipy.org/) >= 1.2
 * [scipy](http://www.scipy.org/) >= 0.7.0
 * [scikit-image](https://github.com/scikit-image/scikit-image)
 * [nibabel](https://github.com/nipy/nibabel) >= 1.2
 * [neuropythy](https://github.com/noahbenson/neuropythy) >= 0.1
 * [pysistence](https://pythonhosted.org/pysistence/) >= 0.4.0
 * [decorator](https://github.com/micheles/decorator) >= 4.0.0

## License #########################################################################################
This README file is part of the SCO Python library.

The SCO Python library is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Usage

You can either run `sco` from the command line or within a python
interpreter. Note that if the defaults are different depending on
which you do!

In either case, the chain runs one subject at a time on any number of
stimulus images, with one set of parameter values.

## Command line

`sco <subject> <image0000> <image0001>...`

Runs the Standard Cortical Observer prediction routine and exports a
series of

MGZ files: `prediction_0000.mgz`, `prediction_0001.mgz`, etc.

The following options may be given:

  * `-h`|`--help` prints this message.
  
  * `-p`|`--deg2px=<value>` sets the pixels per degree in the input
    images (default: 24).
	
  * `-e`|`--max-eccen=<value>` sets the maximum eccentricity modeled
    in the output (default: 20).
	
  * `-o`|`--output=<directory>` sets the output directory (default:
    `.`).

## From within Python

It can also be run within Python (via an interpreter or Jupyter
notebook), in which case you will have greater control over the inputs
and which parts of the chain are run.

To do this, run the following:

```
import sco
results = sco.calc_sco(subject='...', stimulus_image_filenames=['...'])
```

`subject` and `stimulus_image_filenames` are the only parameters that
**must** be set, but there are many parameters that can be tweaked
(see [User-specified parameters](#user-specified-parameters)).

Afterwards, `results` will be a dictionary containing the data pool of
the calc chain, all of the inputs and outputs of the various
functions. See [Data pool](#data-pool) for more details on their
values. The key you will most likely be interested in is
`predicted_responses`, which contains the predicted response of each
voxel to each image.

In order to get an mgh for viewing the predictions on a brain, call
`results=sco.export_predicted_responses(results, export_path='...')`,
after filling out `results` above and specifying where you want to
export the image (this step can be added to the calc chain if you
wish, but it is not by default if run through the interpreter). This
will take the `predicted_responses` key from `results` and create a
corresponding mgh image, which you can view using `freesurfer` or
`nibabel`.


# Data pool

The model has a large data pool (found in the `results` dictionary),
which I describe below. The data pool contains the inputs and outputs
for every step of the model.

For these descriptions, we assume your subject
has n voxels and you're predicting the responses for m images.

## User-specified parameters

* `subject`: `neuropythy.freesurfer.subject` or string. Specifies the
  subject whose anatomical data will be loaded in to determine the pRF
  coordinates and sizes. User-specified with no default value and so
  must be set by the user. In
  `anatomy.core.import_benson14_from_freesurfer`, subject is converted
  from a string to a freesurfer subject (if it was set as a string),
  allowing the model to find the data for the subject.

* `stimulus_image_filenames`: list of strings with length m. Each
  value specifies the file name of one image to be used as a stimulus
  for the model. User-specified with no defaults. Used in
  `stimulus.core.import_stimulus_images`.

* `min_cycles_per_degree`: integer. The lowest frequency (per degree)
  of any of the wavelets. User-defined with a default value of 0. Used
  in `stimulus.core.calc_gabor_filters`.
  
* `wavelet_octaves`: integer, specifies how many octaves (doublings of
  frequency) the preferred spatial frequencies of the Gabor wavelets
  should cover. User-specified, default value of 4. Used in
  `stimulus.core.calc_gabor_filters`.

* `wavelet_steps`: integer, specifies how many steps between doublings
  the preferred spatial frequencies of the Gabor wavelets should
  have. User-specified, default value of 2. Used in
  `stimulus.core.calc_gabor_filters`.

* `max_eccentricity`: `None` or integer. Specifies the max
  eccentricity (degree distance from center of visual field) of the
  pRF for voxels to consider. User-specified, default value is
  `None`. If `None`, value of 90 degrees is used. Used in
  `stimulus.core.calc_gabor_filters`.
  
* `normalized_pixels_per_degree`: integer. The number of pixels per
  degree in the normalized image. User-defined variable, default value
  of 15. First used in
  `stimulus.core.calc_normalized_stimulus_images`.

* `gabor_orientations`: integer. The number of preferred orientations
  the Gabor filters should have. User-specified, with a default value
  of 4. First used in `stimulus.core.calc_gabor_filters.`.

* `stimulus_edge_value`: float. The value outside the projected
  stimulus. Necessary to avoid edge effects with the convolution
  operations. User-defined, with default value of `0.0`. Used in
  `stimulus.core.calc_normalized_stimulus_images`. 0 corresponds to
  the central value, since the images will be 0-centered (running from
  -.5 to .5) before the convolution.

* `stimulus_pixels_per_degree`: integer or list of integers. The
  pixels per degree for the stimulus. User-specified, with a default
  value of 24. First used in
  `stimulus.core.calc_normalized_stimulus_images`. If it's a list,
  must have m values, one for each stimulus image (allowing the images
  to have differing pixels per degree).

* `normalized_stimulus_size`: 2-tuple of integers. Describes the
  target width and depth of the stimulus image in
  pixels.User-specified, with a default value of `(300,300)`. First
  used in `stimulus.core.calc_normalized_stimulus_images`
  
* `Kay2013_pRF_sigma_slope`: dictionary whose keys are 1, 2, and 3 and
  whose values are the ?. User-defined, with default value of `{1:
  0.1, 2: 0.15, 3: 0.27}`. First used in
  `anatomy.core.calc_Kay2013_pRF_sizes`.
  
* `Kay2013_SOC_constant`: dictionary whose keys are 1, 2, and 3 and
  whose values are the second-order contrast constant `c` from Kay et
  al, 2013 for areas V1, V2, and V3. User-defined, with default value
  of `{1: 0.93, 2: 0.99, 3: 0.99}`. First used in
  `normalization.core.calc_Kay2013_SOC_normalization`. Currently, one
  constant is defined per area, but this could be extended so each
  voxel has a separate value.
  
* `Kay2013_output_nonlinearity`: dictionary whose keys are 1, 2, and 3
  and whose values are the compressive nonlinearity constants `n` from
  Kay et al, 2013 for areas V1, V2, and V3. That is, the response is
  raised to this power as the final step in the model.. User-defined,
  with default value of `{1: 0.18, 2: 0.13, 3: 0.12}`. First used in
  `anatomy.core.calc_Kay2013_pRF_sizes`. Currently, one constant is
  defined per area, but this could be extended so each voxel has a
  separate value.

## Created by model

* `v123_labels_mgh`: freesurfer MGH image of entire brain. The values
  are 0 everywhere but V1, V2, and V3, where the values are 1, 2, and
  3, respectively. Loaded in from disk by
  `anatomy.core.import_benson14_from_freesurfer`, this is assumed to
  have already been calculated for the subject by the time the model
  is run.

* `polar_angle_mgh`: freesurfer MGH image of entire brain. The values
  are 0 everywhere but V1, V2, and V3, where the values represent the
  preferred polar angle of the corresponding receptive field. Loaded
  in from disk by `anatomy.core.import_benson14_from_freesurfer`, this
  is assumed to have already been calculated for the subject by the
  time the model is run.

* `ribbon_mghs`: tuple containing two freesurfer MGH images of entire
  brain. The values are 0 everywhere but the border between white and
  grey matter, where they're 1. Each image in the tuple shows border
  for one hemisphere. Loaded in from disk by
  `anatomy.core.import_benson14_from_freesurfer`, this is assumed to
  have already been calculated for the subject by the time the model
  is run.

* `eccentricity_mgh`: freesurfer MGH image of entire brain. The values
  are 0 everywhere but V1, V2, and V3, where the values represent the
  preferred eccentricity. Loaded in from disk by
  `anatomy.core.import_benson14_from_freesurfer`, this is assumed to
  have already been calculated for the subject by the time the model
  is run.

* `pRF_v123_labels`: 1 dimensional numpy array with n entries. Each
  entry is either 1, 2, or 3, specifying whether the corresponding
  voxel is part of V1, V2, or V3. Values are pulled from
  `v123_labels_mgh` in
  `anatomy.core.calc_pRFs_from_freesurfer_retinotopy`.

* `pRF_polar_angle`: 1 dimensional numpy array with n entries. Each
  value represents the preferred polar angle (in radians) of each
  voxel. Values are pulled from `polar_angle_mgh` in
  `anatomy.core.calc_pRFs_from_freesurfer_retinotopy`.

* `pRF_eccentricity`: 1 dimensional numpy array with n entries. Each
  value represents the preferred eccentricity of each voxel. Values
  are pulled from `eccentricity_mgh` in
  `anatomy.core.calc_pRFs_from_freesurfer_retinotopy`.

* `pRF_centers`: n x 2 numpy array giving the x, y centers of the pRFs
  for each voxel. Calculated by
  `anatomy.core.calc_pRFs_from_freesurfer_retinotopy` based on
  eccentricity image.

* `pRF_hemispheres`: 1 dimensional numpy array with length n
  specifying which hemisphere the corresponding voxel is in. 1
  specifies the left hemisphere, -1 the right. Created in
  `anatomy.core.calc_pRFs_from_freesurfer_retinotopy`.

* `pRF_voxel_indices`: n x 3 numpy array. Contains the x, y, and z
  coordinates of each voxel in the brain. Calculated by
  `anatomy.core.calc_pRFs_from_freesurfer_retinotopy`.

* `pRF_sizes`: list with length n, giving the sizes of the pRFs (in
  degrees?) for each voxel. Calculated by
  `anatomy.core.calc_Kay2013_pRF_sizes` based on the eccentricity and
  visual area of the voxel.
  
* `stimulus_images`: list of numpy arrays with length m. Each value is
  the array for one of the `stimulus_image_filenames`. Can be
  visualized with `matplotlib.pyplot.imshow`, these are loaded in from
  `stimulus_image_filenames` in
  `stimulus.core.import_stimulus_images`.

* `normalized_stimulus_images`: list of length m, with each entry
  contained a normalized image. This is calculated by
  `stimulus.core.calc_normalized_stimulus_image`, which normalizes
  each image to the same resolution and size; it zooms in on each
  image so that the pixels per degree is the right value and then is
  cropped so it's the correct size.

* `orientations`: 1d numpy array with number of entries equal to the
  integer specified in `gabor_orientations`. These are the actual
  angle values (in radians) that the Gabor filters have for their
  preferred orientations. Calculated (based on `gabor_orientations`)
  in `stimulus.core.calc_gabor_filters`.

* `wavelet_frequencies`: 1 dimensional numpy array whose number of
  elements is specified by the user-specified number of octaves
  (`wavelet_octaves`) and steps (`wavelet_steps`). Calculated in
  `stimulus.core.calc_gabor_filters`, specifies the preferred spatial
  frequencies (in pixels?) of the Gabor filters used in the first step
  of the model. The frequencies run from `min_cycles_per_degree` to
  `2^wavelet_octaves`, with `wavelet_steps` steps between each
  doubling (there are therefore `wavelet_octaves`*`wavelet_steps`+1
  total frequencies). They are then converted into pixels (from
  degrees).
  
* `filters`: numpy array with all the Gabor filters. The exact
  dimensionality depends on the values of `orientations`,
  `wavelet_frequencies`; the dimensionality will be
  `len(orientations)` x `len(wavelet_frequencies)` (with default
  values, 4 x 9). This way, the filters tile the specified
  orientations and spatial frequencies. Calculated in
  `stimulus.core.calc_gabor_filters`, these are the filters used in
  the first step of the model. The Gabors themselves are created by a
  call to `skimage.filters.gabor_kernel`.
  
* `filtered_images`: list of numpy arrays with length m. Each value is
  the array for a filtered image. That is, it's an image from
  `normalized_stimulus_images` that has had the `filters` convolved
  with it. This is done by `stimulus.core.calc_filtered_images`.

* `pRF_pixel_centers`: n x 2 numpy array. Contains the x, y positions
  of the centers of each voxel's pRF in the visual field. Calculated
  in `pRF.core.calc_pRF_responses`.
  
* `pRF_pixel_sigmas`: list with length n, giving the sigmas (in
  pixels) for the pRFs of each voxel. Calculated by
  `pRF.core.calc_pRF_responses` based on `pRF_sizes`.
  
* `pRF_responses`: m x n numpy array. Contains the predicted responses
  of each voxel's pRF to each image. This is the output of the
  "spatial summation" step from Kay et al, 2013 and is the input to
  the second order contrast step. It's calculated by
  `pRF.core.calc_pRF_responses`.

* `SOC_normalized_responses`: m-length list of 1 dimensional numpy
  arrays with n entries. The responses of each voxel after being put
  through the the second-order contrast calculation: `(x - c *
  x_bar)^2`, where `x_bar` is the average response across voxels to
  this image and `c` is the value of `Kay2013_SOC_constant` for this
  area. Each entry in the list is the predicted response of all voxels
  to one image. This is calculated in
  `normalization.core.calc_Kay2013_SOC_normalization`.

* `predicted_responses`: m x n numpy array. Contains the final
  predicted response of each voxel to each image (so each step has
  been applied, the final one is the output nonlinearity). Calculated
  in `normalization.core.calc_Kay2013_output_nonlinearity`.

# References

- Kay, K. N., Winawer, J., Rokem, A., Mezer, A., & Wandell,
  B. A. (2013). A two-stage cascade model of BOLD responses in human
  visual cortex. {PLoS} Comput Biol,
  9(5), 1003079. http://dx.doi.org/10.1371/journal.pcbi.1003079
