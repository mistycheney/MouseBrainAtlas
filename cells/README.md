`run_third_party_tools_wholestack_gordon_parallel`: run cellprofiler or farsight, generate cell label map.
`detect_cells`: extract cell info.
`align_pad_mirror_cells`: align cells, pad to the same size, generate other three mirroring versions.
`compute_features`: compute features. The core function `compute_features_regions()` is defined in `cell_utilities.py`.
`train_classifiers_cell_based`:

`spectral_embedding_size_normalized_v2`: spectral embedding of shape
One section has ~100k cells.

`reconstruct_images`: reconstruct images

# Cell-based Features #

Section-wise data:
`detected_cells/<stack>/<fn>/<fn>_<what>.<ext>`

Cell orientations:
`blobOrientations.bp`

Mirror directions:
`cells_aligned_mirrorDirections.bp`

Orientation-normalized cell blobs:
`cells_aligned_mirrored_padded.bp`

Indices of large cells larger than 30 um^2 (indices into the list `cells_aligned_mirrored_padded`)
`largeCellIndices.bp`

Neighborhood relationship
`neighbor_info.pkl`
This is a dict with four keys:
`neighbors`: dict of {cell index: neighbors sorted by distance}
`neighbor_vectors`: dict of {cell index: list of vectors from the cell to each neighbor, sorted by distance}
`radial_indices`: dict of {cell index: radial histogram}
`angular_indices`: dict of {cell index: angular histogram}


# Morphology Features #

Large cells are selected. Their indices are `largeCellIndices.bp`. Their features `largeCellFeatures.bp` include:
- 0: orientation
- 1: mirror direction (0,1,2 or 3)
- 2: size
- 3-9: Hu moments

# Connection Features #
Neighbor are cells with a distance of 50 um.
- `neighborCellIndices.hdf`: dict of {large cell index: neighbor indices sorted by distance (incl. both large and small cells)}
- `neighborVectors.hdf`: dict of {large cell index: neighbor vectors}
- `neighborRadialHistBins.hdf`: dict of {large cell index: each neighbor's radial histogram bins}
- `neighborAngularHistBins.hdf`: dict of {large cell index: each neighbor's angular histogram bins}

# Region Features #
For a given region described by contour vertex coordinates,
['largeSmallLinkDirHist',
 'largeSizeWeightedHist',
 'largeLargeLinkLenHist',
 'largeLargeLinkSizeDiffHist',
 'largeSizeHist',
 'largeSmallLinkLenHist',
 'allSizeHist',
 'largeOrientationHist',
 'largeLargeLinkOrientationDiffHist',
 'largeLargeLinkDirHist',
 'largeLargeLinkJacHist',
 'huMomentsHist']
