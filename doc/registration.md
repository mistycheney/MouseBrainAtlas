# Processing a new stack given a trained atlas

## Compute the features at sparse locations on each image.
`learning/compute_features_for_entire_stacks.ipynb`

## Convert the image stack to 3-D probability maps.
`learning/from_images_to_score_volume.ipynb`

## Register to atlas.

### Simple global registration
`registration/registration_v7_atlasV6_simpleGlobal`

### Local registration

Structures are further adjusted either individually or in groups.

`registration/registration_v7_atlasV6_local_allstructures`

`$ ./register.py <transform_spec>`
