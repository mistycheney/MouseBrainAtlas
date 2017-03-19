`run_third_party_tools_wholestack_gordon_parallel`: run cellprofiler or farsight, generate cell label map.
`detect_cells`: extract cell info.
`align_pad_mirror_cells`: align cells, pad to the same size, generate other three mirroring versions.
`compute_features`: compute features. The core function `compute_features_regions()` is defined in `cell_utilities.py`.
`train_classifiers_cell_based`:

`spectral_embedding_size_normalized_v2`: spectral embedding of shape
One section has ~100k cells.

`reconstruct_images`: reconstruct images


# Hu moments #

Large cells are selected. Their indices are `largeCellIndices.bp`. Their features `largeCellFeatures.bp` include:
- orientation
- mirror direction (0,1,2 or 3)
- size
- Hu moments
