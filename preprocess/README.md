This folder contains code related to preprocessing (aligning, cropping, mask generation).

Preprocessing is supposed to be done with the GUI `/gui/preprocess_tool_v2.py`.

The preprocessing involves the following steps:
- Assign slide, position to filename.
  It can be any of a valid filename, rescan, placeholder, nonexisting.
  - Rescan are slides that look valid on macro but no corresponding image is received. Now they are basically identical to placeholder since they are unlikely going to be scanned and re-sent.
- Sort
- Align
  - server Align
  - download output images resulted from each consecutive transform. `<stack>_elastix_output/*/result.0.tif`
  - (user) check consecutive alignments. In this case, the user should manually specify matching point pairs based on which program will compute the transform.
- Compose
  - server compose
  - download folder `<stack>_thumbnail_unsorted_alignedTo_<anchor_fn>` and the list of composed transforms `<stack>_transformsTo_<anchor_fn>.pkl`.
  - (user) check composed alignment result. Adjust section order.
- Crop
  - (user) specify crop box and FIRST, LAST
  - generate `<stack>_thumbnail_unsorted_alignedTo_<anchor_fn>_cropped` and `<stack>_lossless_unsorted_alignedTo_<anchor_fn>_cropped`; also `<stack>_lossless_unsorted_alignedTo_<anchor_fn>_cropped_saturation` and `<stack>_lossless_unsorted_alignedTo_<anchor_fn>_cropped_compressed`
  (generating saturation or compressed versions take 10 minutes)

### Generate Auxiliary versions ###

* A JPEG version for exporting visualizations. `_compressed`.
* A grayscale version for input to DNN. `_grayscale`


- Generate mask
  - server gen mask
  notebook `generate_mask_entropy.ipynb`
Note: For fluorescent images, red and green channels give higher contrast. For nissl images, blue channel is better.
Note: Superpixel is necessary because it is eaaier to thresholded image
Snake must be done on original or contrast enhanced, not thresholded because thresholded will be jagged.

  tune TRAIN_DISTANCES_AS_DISTANCE_PERCENTILE
  - download folder `<stack>_mask_unsorted` and visualization `<stack>_maskContourViz_unsorted`
  1500 seconds
  `learn_to_identify_wrong_masks`
  - (user) check mask contoured images

- Warp crop mask
  - server warp crop masks
  - download folder `<stack>_mask_unsorted_alignedTo_<anchor_fn>` and ``<stack>_mask_unsorted_alignedTo_<anchor_fn>_cropped``

- Sync to workstation

- Confirm order
  - create sorted versions of results by creating symbolic links to files in unsorted folders
