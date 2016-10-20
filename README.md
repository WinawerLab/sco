# sco ##############################################################################################
The Standard Cortical Observer Python library.

## Author ##########################################################################################

 * [Noah C. Benson](https://github.com/noahbenson) &lt;<nben@nyu.edu>&gt; (*corresponding author*)
 * [Bill Broderick](https://github.com/billbrod
 * [Catherine Olson](https://github.com/catherio)
 * [Heiko Mueller](https://github.com/heikomuller)
 * [Jonathan Winawer](https://github.com/jwinawer)

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

Any parameter that is specified as an array can also be specified as a
list. However, the model will turn it into an array for ease of
handling. For the parameters that can be set as a single value or an
array, if a single value is specified by the user, the model will cast
it as an array (with every entry the same) for ease of use.

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

* `stimulus_aperture_edge_width`: integer or 1-dimensional array of m
  integers. User-defined, can be set as a single integer (in which
  case all images will have the same value) or an array of integers
  (in which case image i will use `stimulus_aperture_edge_width[i]`);
  default is for all to have the same value, which is equal to the
  `normalized_pixels_per_degree`. This gives the number of pixels over
  which the aperture should be smoothly extended; 0 gives a hard edge,
  otherwise a half-cosine smoothing is used. Used in
  `stimulus.core.image_apply_aperture`.

* `max_eccentricity`: integer or 1-dimensional array of m
  integers. Specifies the max eccentricity (degree distance from
  center of visual field) of the pRF for voxels to
  consider. User-specified, default value is 12 or
  `normalized_stimulus_aperture / normalized_pixels_per_degree` if
  both are set. Used in
  `anatomy.core.calc_pRFs_from_freesurfer_retinotopy`. If an integer,
  same value will be used for each image; if an array, each image will
  use its corresponding value from the array.
  
* `normalized_pixels_per_degree`: integer or 1-dimensional array of m
  integers. The number of pixels per degree in the normalized
  image. User-defined variable, default value of 15 or
  `max_eccentricity * normalized_stimulus_aperture` if both of those
  are set. First used in
  `stimulus.core.calc_normalized_stimulus_images`. If an integer, same
  value will be used for each image; if an array, each image will use
  its corresponding value from the array.
  
* `normalized_stimulus_aperture`: integer or 1-dimensional array of m
  integers. The radius (in pixels) of the aperture to apply after each
  image has been normalized in order to get the reduced view
  corresponding to the models input. User-defined, default value is
  `max_eccentricity * normalized_pixels_per_degree`. If an integer,
  same value will be used for each image; if an array, each image will
  use its corresponding value from the array.

* `gabor_orientations`: integer. The number of preferred orientations
  the Gabor filters should have. User-specified, with a default value
  of 8. First used in `contrast.core.calc_stimulus_contrast_functions`.

* `stimulus_edge_value`: float. The value outside the projected
  stimulus. Necessary to avoid edge effects with the convolution
  operations. User-defined, with default value of `0.0`. Used in
  `stimulus.core.calc_normalized_stimulus_images`. 0 corresponds to
  the central value, since the images will be 0-centered (running from
  -.5 to .5) before the convolution.

* `stimulus_pixels_per_degree`: integer or 1-dimensional array of
  integers. The pixels per degree for the stimulus. User-specified,
  with a default value of 24. First used in
  `stimulus.core.calc_normalized_stimulus_images`. If it's an array,
  must have m values, one for each stimulus image (allowing the images
  to have differing pixels per degree).

* `normalized_stimulus_size`: 2-tuple of integers. Describes the
  target width and depth of the stimulus image in
  pixels.User-specified, with a default value of `(300,300)`. First
  used in `stimulus.core.calc_normalized_stimulus_images`

* `pRF_blob_stds`: a positive number specifying the number of standard
  deviations to include in the pRF Gaussian blob definition. By default
  this value is 2. Used in the `pRF.core.calc_pRF_matrices function`.
  
All the Kay2013 parameters can be a single float (in which case the
same value is used for each voxel), a list/array of floats (which must
be length n, in which case each voxel will use the corresponding
value), or a dictionary with 1, 2 and 3 as its keys and with floats as
the values (specifying the values for these parameters for voxels in
areas V1, V2, and V3).
  
* `Kay2013_pRF_sigma_slope`: float, array, or dictionary whose keys
  are 1, 2, and 3 and whose values give the slope of the linear
  relationship between the eccentricity of a pRF and its
  size. User-defined, with default value of `{1: 0.1, 2: 0.15, 3:
  0.27}`, values from Kay2013b. First used in
  `anatomy.core.calc_Kay2013_pRF_sizes`.

* `Kay2013_output_nonlinearity`: float, array, or dictionary whose
  keys are 1, 2, and 3 and whose values are the compressive
  nonlinearity constants `n` from Kay2013a for areas V1, V2, and
  V3. That is, the response is raised to this power as the final step
  in the model. User-defined, with default value of `{1: 0.18, 2:
  0.13, 3: 0.12}`. First used in
  `anatomy.core.calc_Kay2013_pRF_sizes`.
  
* `Kay2013_normalization_r`: float, array, or dictionary with keys 1,
  2, and 3. Value is the the normalization parameter `r` from
  Kay2013a, which (along with `Kay2013_normalization_s`) controls the
  strength of the divisive normalization step. User-specified with
  default value of `1`. Used in
  `contrast.core.calc_divisive_normalization_functions`.

* `Kay2013_normalization_s`: float, array, or dictionary with keys 1,
  2, and 3. Value is the the normalization parameter `s` from
  Kay2013a, which (along with `Kay2013_normalization_r`) controls the
  strength of the divisive normalization step. User-specified with
  default value of `.5`. Used in
  `contrast.core.calc_divisive_normalization_functions`.
  
* `Kay2013_SOC_constant`: float, array, or dictionary whose keys are
  1, 2, and 3 and whose values are the second-order contrast constant
  `c` from Kay2013a for areas V1, V2, and V3. User-defined, with
  default value of `{1: 0.93, 2: 0.99, 3: 0.99}`. First used in
  `normalization.core.calc_Kay2013_SOC_normalization`.
  
* `Kay2013_response_gain`: float, array, or dictionary whose keys
  are 1, 2, and 3. Value is the response gain for that voxel, paramter
  `g` from Kay2013a. User-defined, with default value of `1`. Used in
  `normalization.core.calc_Kay2013_output_nonlinearity`.

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

* `pRF_sizes`: 1 dimensional numpy array with length n, giving the
  sizes of the pRFs (in degrees?) for each voxel. Calculated by
  `anatomy.core.calc_Kay2013_pRF_sizes` based on the eccentricity and
  visual area of the voxel.
  
* `stimulus_images`: array of numpy arrays with length m. If the
  `stimulus_images` are different sizes, which is likely, this array
  will have `dtype=object`, so it will act as a one-dimensional array
  containing more arrays. If they're the same size, it will be a 3d
  array with indices corresponding to image, pixel x, and pixel
  y. Each value is the array for one of the
  `stimulus_image_filenames`. Can be visualized with
  `matplotlib.pyplot.imshow`, these are loaded in from
  `stimulus_image_filenames` in
  `stimulus.core.import_stimulus_images`.

* `normalized_stimulus_images`: 1-dimensional array of length m, with
  each entry contained a normalized image. This is calculated by
  `stimulus.core.calc_normalized_stimulus_image`, which normalizes
  each image to the same resolution and size; it zooms in on each
  image so that the pixels per degree is the right value and then is
  cropped so it's the correct size.
  
* `stimulus_contrast_functions`: 1-dimensional array of m
  functions. Each function corresponds to one image, takes in a
  frequency and returns an image that has been transformed from the
  original `normalized_stimulus_images` to a new image the same size
  in which each pixel represents the contrast energy at that point and
  at the given frequency. Equivalent to convoluting the image with a
  filter bank with that spatial frequency preference. The function
  also caches its results for a given frequency, to reduce calculation
  time. Created in `contrast.core.calc_stimulus_contrast_functions`.
  
* `normalization_contrast_functions`: 2-dimensional array, n x m, with
  a separate function for each voxel for each image. Each function
  takes in a frequency, calls the `stimulus_contrast_function` for
  that image and spatial frequency, and then divisively normalizes
  that image with the corresponding `Kay2013_normalization_r` and
  `Kay2013_normalization_s` for that voxel. The function also caches
  its result for a given frequency, r, and s for speed. Created in
  `contrast.core.calc_divisive_normalization_functions`.

* `pRF_pixel_centers`: n x m x 2 numpy array. Contains the x, y
  positions of the centers of each voxel's pRF in the visual
  field. Differs from `pRF_centers` because different images may have
  different pixels per degree, so we have a separate value for each
  image. Calculated in `pRF.core.calc_pRF_pixel_data`, based on
  `pRF_centers`.
  
* `pRF_pixel_sizes`: n x m numpy array, giving the sizes (in
  pixels) for the pRFs of each voxel. Differs from `pRF_sizes` because
  different images may have different pixels per degree, so we have a
  separate value for each image. Each entry gives the pRF size for a
  given image in a given voxel. Calculated by
  `pRF.core.calc_pRF_pixel_data` based on `pRF_sizes`.
  
* `pRF_frequency_preference_function`: a function that takes the
  eccentricity, pRF size, and visual label and returns a dictionary
  whose keys are frequencies (in cycles per degree) and whose values
  are the weights applied to that particular frequency at the given
  eccentricity, pRF size, and visual area. Can be user defined, but
  also has default value created in
  `pRF.core.calc_pRF_defualt_options`. Used in
  `pRF.core.calc_pRF_responses`.

* `pRF_matrices`: n x m numpy array of image matrices where n is the
  number of pRFs and m is the number of images. Each element of the
  pRF_matrices value is a sparse array (scipy.sparse.csr_matrix) the
  same size as the normalized stimulus image to which it corresponds;
  the values in this image correspond to weights on the pRF Gaussian
  and are set to sum to 1. This is calculated by
  pRF.core.calc_pRF_matrices.
  
* `SOC_responses`: n x m numpy array. The responses of each voxel after
  being put through the the second-order contrast calculation:
  `(x - c * x_bar)^2`, where `x_bar` is the weighted average contrast
  value (see normalized_contrast_functions) across pixels in the pRF
  and `c` is the value of `Kay2013_SOC_constant` for this area. Each entry
  in the list is the predicted response of all voxels to one image. This
  is calculated in `normalization.core.calc_Kay2013_SOC_normalization`.

* `predicted_responses`: m x n numpy array. Contains the final
  predicted response of each voxel to each image (so each step has
  been applied, the final one is the output nonlinearity). Calculated
  in `normalization.core.calc_Kay2013_output_nonlinearity`.

# References

- Kay2013a: Kay, K. N., Winawer, J., Rokem, A., Mezer, A., & Wandell,
  B. A. (2013). A two-stage cascade model of BOLD responses in human
  visual cortex. {PLoS} Comput Biol,
  9(5), 1003079. http://dx.doi.org/10.1371/journal.pcbi.1003079

- Kay2013b: Kay, K. N., Winawer, J., Mezer, A., & Wandell,
  B. A. (2013). Compressive spatial summation in human visual
  cortex. Journal of Neurophysiology, 110(2),
  481–494. http://dx.doi.org/10.1152/jn.00105.2013
