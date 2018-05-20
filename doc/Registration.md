# Processing a new stack given a trained atlas

## Compute the features at sparse locations on each image.
`learning/compute_features_for_entire_stacks.ipynb`

## Simple global registration
Use the GUI to select the 2D center of 12N and 3N(either L or R). Then
`registration/registration_v7_atlasV6_simpleGlobal`

This helps reduce the computational cost for the following detection step.

## Convert the image stack to 3-D probability maps.
`learning/from_images_to_score_volume.ipynb`

## Local registration

Structures are further adjusted either individually or in groups.

`registration/registration_v7_atlasV6_local_allstructures`

`$ ./register.py <transform_spec>`


# Transform parameters

A transform can be expressed in any of the following ways:

* dictionary
  - `parameters`: 12-array, flattened version of the rigid or affine 3x4 matrix.
  - `centroid_m_wrt_wholebrain`: 3-array, initial shift of the moving volume, relative to the wholebrain origin.
  - `centroid_f_wrt_wholebrain`: 3-array, initial shift of the fixed volume, relative to the wholebrain origin.
* (4,4) matrix: the 4x4 matrix that represents the transform.
* (3,4) matrix: first three rows of the full 4x4 matrix.
* (12,) array: flattened array of the first three rows of the full 4x4 matrix.

For each registration, the following results are stored:
- `<registration_identifier>_parameters.json`: contains three keys `centroid_f_wrt_wholebrain`((3,)-array), `centroid_m_wrt_wholebrain`((3,)-array) and `parameters`((12,)-array).
- `<registration_identifier>_scoreHistory.bp`: the score history as a list
- `<registration_identifier>_scoreEvolution.png`: plot of the score over iterations
- `<registration_identifier>_trajectory.bp`: trajectory of the parameters during optimization, a list of 12 parameters.

Mathematically, a transform is expressed as:

q - q0 = R * T0(p-p0) + t

- `p` is a point in moving brain (wrt wholebrain in the case of a subject brain, or canonicalAtlasSpace in the case of atlas)
- `p0` is the rotation center defined on moving brain.
- `q` is a point in fixed brain (wrt wholebrain)
- `q0` is the shift of fixed brain.
- `R`, the 3x3 rotation matrix, which is part of the computed transform.
- `t`, 3-array, which is part of the computed transform.
- `T0` is the initial transform. 

# Using `Aligner` class

- Specify `T0` with `Aligner.set_initial_transform`.
- Specify `p0` and `q0` with `Aligner.set_centroids`
- Optionally, `Aligner.do_grid_search`.
- Run `Aligner.optimize` to compute `R` and `t`.
- Compose initial transform and estimated transform using `compose_alignment_parameters`.
