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
