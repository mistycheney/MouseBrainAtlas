# Detection

## Compute the features on a grid for every image.

Run `learning/compute_features.py <brain_name> [--section <section_number>]`

Reference: `learning/compute_features_for_entire_stacks.ipynb`

## Simple global registration

This step perform a simple texture-independent registration that roughly aligns the atlas to the subject.
The purpose is to find a small 3-D extent (compared to the full subject) to run the subsequent detection.

This can be achieved by any of the following:
- Use the GUI to select a set of anchor points.
For example, we can use two points: the centroid of 12N and 3N(either L or R). Then
`registration/registration_v7_atlasV6_simpleGlobal`
- Automatically align the 3-D section outline.

## Convert the image stack to 3-D probability maps.
Run `learning/from_images_to_score_volumes.py <brain_name> <detector_id> --structure_list <json-encoded list str>`

Reference: `learning/from_images_to_score_volume.ipynb`
