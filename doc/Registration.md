# Registration specs

A _registration_spec_ specifies:
  - stack_m: dict, moving brain spec
  - stack_f: dict, fixed brain spec
  - warp_setting: int, registration setting id

# Brain specs

A _brain_spec_ specifies:
- name: brain name
- vol_type: volume type
- structure: str or list. If list, these structures are transformed as an integral group.
- resolution: 
- detector_id: mandatory if vol_type is "score".

See `example_fixed_brain_spec.json` for an example.

# Registration settings
A set of registration settings are defined in `registration/registration_settings.csv`.
Each setting specifies the following parameters:
- warp_id
- upstream_warp_id
- transform_type: rigid or affine, or bspline
- grad_computation_sample_number
- grid_search_sample_number
- std_tx_um
- std_ty_um
- std_tz_um
- std_theta_xy_degree
- surround_weight
- regularization_weight
- terminate_thresh_trans
- terminate_thresh_rot
- history_len
- max_iter_num
- learning_rate_trans
- learning_rate_rot
- comment


# Local registration

Run `registration/register_brain.py <fixed_brain_spec_json> <moving_brain_spec_json> <registration_setting_id> [--use_simple_global]`

Generated registration results are stored at
`CSHL_registration_parameters/<atlas_name>/<atlas_name>_10.0um_scoreVolume_<moving_structures>_warp<registration_id>_<fixed_brain>_detector<detector_id>_10.0um_scoreVolume_<fixed_structures>`.
Also see [Explanation of registration results](DataDescription.md)

Transformed moving structures (including the corresponding surround structures) are stored at 
`CSHL_volumes/<atlas_name>/<moving_brain_spec>_warp<registration_setting_id>_<fixed_brain_spec>/score_volumes/`

For each structure, the volume spec is `<moving_brain_spec>_warp<registration_setting_id>_<fixed_brain_spec>_<sided_or_surround_structure>`.
Origin wrt is `origin_wrt_fixedWholebrain.txt`.

To overlay transformed atlas on section images, run 
`registration/visualize_registration.py <fixed_brain_spec_json> <moving_brain_spec_json> <registration_setting_id> [--structure_list <json_encoded_structure_list>]`

Notebook: `registration/registration_v7_atlasV6_local_allstructures.ipynb`

# Using `Aligner` class

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


