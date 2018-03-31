# Transform parameters
- `parameters`: 12-array, flattened version of the rigid or affine 3x4 matrix.
- `centroid_m_wrt_wholebrain`: 3-array, initial shift of the moving volume, relative to the wholebrain origin.
- `centroid_f_wrt_wholebrain`: 3-array, initial shift of the fixed volume, relative to the wholebrain origin.

This can also be expressed in 4 x 4 matrix form using `alignment_parameters_to_transform_matrix_v2`.
