# Detection

## Compute features using a moving window

Run `learning/compute_features.py <brain_name> [--section <section_number>] [--win_id <window_id>]`

Output is under `/data/CSHL_patch_features/inception-bn-blue/<brain>/<brain>_prep2_none_win7`
- `<image_name>_prep2_none_win7_inception-bn-blue_features.bp`: feature vectors. n x 1024 array (float), each row is the feature at one location.
- `<image_name>_prep2_none_win7_inception-bn-blue_locations.bp`: location of each feature. n x 2(?) array (int), each row is (x,y,??)

Notebook: `learning/compute_features_for_entire_stacks.ipynb`

## Simple global registration (optional)

This step performs a simple texture-independent registration that roughly aligns the atlas to the subject.
The purpose is to find a small 3-D extent (compared to the full subject) to run the subsequent detection.

This can be achieved by any of the following:
- Use the GUI to select a set of anchor points.
For example, we can use two points: the centroid of 12N and 3N(either L or R). Then
`registration/registration_v7_atlasV6_simpleGlobal`
- Automatically align the 3-D brain outline.

## Convert the image stack to 3-D probability maps.

Run `learning/from_images_to_score_volumes.py <brain_name> <detector_id> --structure_list <json-encoded list str>`

2-D score map output is under 
- `/home/yuncong/CSHL_scoremaps/10.0um/<brain_name>/<brain_name>_prep2_10.0um_detector<detector_id>/<image_name>_prep2_10.0um_detector<detector_id>/<image_name>_prep2_10.0um_detector<detector_id>_<structure>_scoremap.bp`. 2-d probability map for one classifier (float between 0 and 1).
- `/home/yuncong/CSHL_scoremap_viz/10.0um/<structure>/<brain_name>/detector<detector_id>/prep2/<image_name>_prep2_10.0um_<structure>_detector<detector_id>_scoremapViz.jpg`. Section image with probability map overlay.

3-D score map output is under
- `/home/yuncong/CSHL_volumes/<brain_name>/<brain_name>_detector<detector_id>_10.0um_scoreVolume`
  - `score_volumes`. Score volume; 3-d float array.
    - volume spec: `<brain_name>_detector<detector_id>_10.0um_scoreVolume_<structure>`.
  - `score_volume_gradients`.  Gradients of score volume. 3 x 3-d float array.
    - volume spec: `<brain_name>_detector<detector_id>_10.0um_scoreVolume_<structure>_gradients`. **(TO FIX: "gradients" is missing from gradient origin filenames)**

Notebook: `learning/from_images_to_score_volume.ipynb`
