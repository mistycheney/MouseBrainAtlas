# Transform parameters

This is expressed in either of the following ways:

One:
- `parameters`: 12-array, flattened version of the rigid or affine 3x4 matrix.
- `centroid_m_wrt_wholebrain`: 3-array, initial shift of the moving volume, relative to the wholebrain origin.
- `centroid_f_wrt_wholebrain`: 3-array, initial shift of the fixed volume, relative to the wholebrain origin.

Two: in 4 x 4 matrix form using `alignment_parameters_to_transform_matrix_v2`.

For each registration, the following results are stored:
- `<registration_identifier>_parameters.json`: contains three keys `centroid_f_wrt_wholebrain`(3-array), `centroid_m_wrt_wholebrain`(3-array) and `parameters`(12-array).
- `<registration_identifier>_scoreHistory.bp`: the score history as a list
- `<registration_identifier>_scoreEvolution.png`: plot of the score over iterations
- `<registration_identifier>_trajectory.bp`: trajectory of the parameters during optimization, a list of 12 parameters.
