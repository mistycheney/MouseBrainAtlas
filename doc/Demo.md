This demo assumes a subject brain (DEMO999) is roughly globally aligned with the atlas (atlasV7).
It shows how one can:
- register 7N_L (facial motor nucleus) individually.
- register 3N_R and 4N_R as a group.
- visualize the aligned atlas overlaid on original images

---------------------------

## Download input data

Set the environment variable to point to the folder of the downloaded code repository.
- `export REPO_DIR=[repo_dir]`.

Specify a folder `demo_data_dir` to store demo input data.
First run `download_demo_data.py [demo_data_dir]` to download input data into this folder.

Set the environment variables which refer to a root directory that contains both input and output data.
- `export ROOT_DIR=[demo_data_dir]`
- `export DATA_ROOTDIR=[demo_data_dir]`

## Register 12N individually
- `$ ./register_brains_demo.py demo_fixed_brain_spec_12N.json demo_moving_brain_spec_12N.json`

## Register 3N_R and 4N_R as a group
- `$ ./register_brains_demo.py demo_fixed_brain_spec_3N_R_4N_R.json demo_moving_brain_spec_3N_R_4N_R.json`

The program should finish in 2 minutes.

The outputs are also generated in _demo_data_ folder under the following paths. You can download the expected output from our S3 bucket using URLs formed by prepending https://s3-us-west-1.amazonaws.com/mousebrainatlas-data/ to the paths.

**Best set of transform parameters**
- `CSHL_registration_parameters/atlasV7/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_R_4N_R_parameters.json`

**Optimization trajectory of transform parameters**
- `CSHL_registration_parameters/atlasV7/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_R_4N_R_trajectory.bp`

**Score history**
- `CSHL_registration_parameters/atlasV7/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_R_4N_R_scoreHistory.bp`

**Score evolution plot**
- `CSHL_registration_parameters/atlasV7/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_R_4N_R_scoreEvolution.png`

**Simple globally aligned moving brain volumes**
- `CSHL_volumes/atlasV7/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp0_DEMO999_detector799_10.0um_scoreVolume_3N_4N_10.0um/score_volumes/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp0_DEMO999_detector799_10.0um_scoreVolume_3N_R_4N_R_10.0um_3N_R.bp`

**Locally aligned moving brain volume**
- `CSHL_volumes/atlasV7/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N_10.0um/score_volumes/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_R_4N_R_10.0um_3N_R.bp`

------------------------

## Visualize registration results

To visualize the multi-probability level structures of the aligned atlas overlaid on original images:
- `$ ./visualize_registration_demo.py demo_visualization_per_structure_alignment_spec.json -g demo_visualization_global_alignment_spec.json`

The outputs are the following:

**Atlas-overlaid images**
- under `CSHL_registration_visualization/DEMO999_atlas_aligned_multilevel_down16_all_structures/NtbNormalizedAdaptiveInvertedGammaJpeg/`


