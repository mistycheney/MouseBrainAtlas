
## Local registration

Structures are further adjusted either individually or in groups.

`registration/registration_v7_atlasV6_local_allstructures`

`$ ./register.py <transform_spec>`

Also see [Explanation to registration results](FileOrganization.md)

# Using `Aligner` class

- First prepare the registration specification.

## Registration settings

`registration/registration_settings.csv`

## Run registration
- Generate parameters based on the registration specification, using `generate_aligner_parameters_v2`
- Create an `Aligner` object.
- Compute gradients of fixed volumes, using `Aligner.compute_gradient`
- (Optional) Set label weights with `Aligner.set_label_weights`.
- Specify `T0` with `Aligner.set_initial_transform`.
- Specify `p0` and `q0` with `Aligner.set_centroids`.
- Optionally, `Aligner.do_grid_search`.
- Run `Aligner.optimize` to compute `R` and `t`.
- Compose initial transform and estimated transform using `compose_alignment_parameters`.
- Save registration results using `DataManager.save_alignment_results_v3`.

## Apply estimated transforms

- Load registration results using `DataManager.load_alignment_results_v3`.
- Load original moving volumes using `DataManager.load_original_volume_v2`.
- Transform moving volumes using `transform_volume_v4`
- Save transformed volumes using `DataManager.save_transformed_volume_v2`.
