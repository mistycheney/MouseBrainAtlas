# Detection

## Compute the features on a grid for every image.

Run `learning/compute_features.py <brain_name> [--section <section_number>]`

Reference: `learning/compute_features_for_entire_stacks.ipynb`

## Simple global registration
Use the GUI to select the 2D center of 12N and 3N(either L or R). Then
`registration/registration_v7_atlasV6_simpleGlobal`

This helps reduce the area considered by the following detection step.

## Convert the image stack to 3-D probability maps.
`learning/from_images_to_score_volumes.py <brain_name> <detector_id> --structure_list <json-encoded list str>`

Reference: `learning/from_images_to_score_volume.ipynb`
