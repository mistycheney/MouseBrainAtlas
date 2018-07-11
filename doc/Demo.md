To run the registration demo, 
- Run `download_demo_data.py`. Make sure the following files are available
  - demo_fixed_brain_spec.json
  - demo_moving_brain_spec.json
  - `CSHL_volumes/DEMO999/DEMO999_detector799_10.0um_scoreVolume/score_volumes/*`
  - `CSHL_volumes/atlasV7/score_volumes/*`
  - `CSHL_simple_global_registration/DEMO999_T_atlas_wrt_canonicalAtlasSpace_subject_wrt_wholebrain_atlasResol.bp`
- `$ ROOT_DIR=/home/yuncong/Brain/demo_data/ DATA_ROOTDIR=/home/yuncong/Brain/demo_data/  ./register_brains_demo.py demo_fixed_brain_spec.json demo_moving_brain_spec.json 7 -g`

The program should finish in 2 minutes.

The expected outputs are listed below. You can download them from our S3 bucket using URLs formed by prepending https://s3-us-west-1.amazonaws.com/mousebrainatlas-data/ to the paths.

**Best set of transform parameters**
- CSHL_registration_parameters/atlasV7/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N_parameters.json

**Optimization trajectory of transform parameters**
- CSHL_registration_parameters/atlasV7/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N_trajectory.bp

**Score history**
- CSHL_registration_parameters/atlasV7/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N_scoreHistory.bp

**Score evolution plot**
- CSHL_registration_parameters/atlasV7/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N_scoreEvolution.png

**Simple globally aligned moving brain volumes**
- CSHL_volumes/atlasV7/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp0_DEMO999_detector799_10.0um_scoreVolume_3N_4N_10.0um/score_volumes/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp0_DEMO999_detector799_10.0um_scoreVolume_3N_4N_10.0um_3N_R.bp

**Locally aligned moving brain volume**
- CSHL_volumes/atlasV7/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N_10.0um/score_volumes/atlasV7_10.0um_scoreVolume_3N_R_4N_R_warp7_DEMO999_detector799_10.0um_scoreVolume_3N_4N_10.0um_3N_R.bp

------------------------

To visualize the multi-probability level structures of the aligned atlas overlaid on original images:
- make sure the following image files are available:
  - CSHL_data_processed/DEMO999/DEMO999_prep2_raw_NtbNormalizedAdaptiveInvertedGammaJpeg/[imgName]\_prep2_raw_NtbNormalizedAdaptiveInvertedGammaJpeg.jpg
- `$ ROOT_DIR=/home/yuncong/Brain/demo_data/ DATA_ROOTDIR=/home/yuncong/Brain/demo_data/ ./visualize_registration_demo.py demo_fixed_brain_spec.json demo_moving_brain_spec.json 7 --structure_list  "[\"3N_R\", \"4N_R\"]"`

The outputs are the following:

**Atlas-overlaid images**
- CSHL_registration_visualization/DEMO999_atlas_aligned_multilevel_down16_all_structures/NtbNormalizedAdaptiveInvertedGammaJpeg/


