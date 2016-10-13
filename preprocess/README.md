Preprocessing is supposed to be done with the GUI `/gui/preprocess_tool.py`.

The preprocessing involves the following steps:
- Sort
- Align
  - server Align
  - download output images resulted from each consecutive transform.
  - (user) check consecutive alignments. In this case, the user should manually specify matching point pairs based on which program will compute the transform.
- Compose
  - server compose
  - download folder `<stack>_thumbnail_unsorted_alignedTo_<anchor_fn>` and the list of composed transforms `<stack>_transformsTo_<anchor_fn>.pkl`.
  - (user) check composed alignment result. Adjust section order.
- Crop
  - (user) specify crop box and FIRST, LAST
  - generate `<stack>_thumbnail_unsorted_alignedTo_<anchor_fn>_cropped` and `<stack>_lossless_unsorted_alignedTo_<anchor_fn>_cropped`; also `<stack>_lossless_unsorted_alignedTo_<anchor_fn>_cropped_saturation` and `<stack>_lossless_unsorted_alignedTo_<anchor_fn>_cropped_compressed`

- Generate mask
  - server gen mask
  - download folder `<stack>_mask_unsorted` and visualization `<stack>_maskContourViz_unsorted`
  - (user) check mask contoured images
- Warp crop mask
  - server warp crop masks
  - download folder `<stack>_mask_unsorted_alignedTo_<anchor_fn>` and ``<stack>_mask_unsorted_alignedTo_<anchor_fn>_cropped``

- Sync to workstation

- Confirm order
  - create sorted versions of results by creating symbolic links to files in unsorted folders
