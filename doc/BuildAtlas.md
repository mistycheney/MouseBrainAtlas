# Building atlas

## Reconstruct annotation brain from 2-D contours

`annotation/construct_annotation_volume_from_annotation_files_v5.ipynb`

Output are at
`CSHL_meshes/<brain_name>/<brain_name>_10.0um_annotationAsScoreVolume/<brain_name>_10.0um_annotationAsScoreVolume_<sided_or_surround_structure>.stl`

## Average annotation volumes to build atlas

Reference: `3d/build_atlas_from_aligned_annotated_brains_v6.ipynb`

### Compute average positions

- Mean positions (wrt _canonicalAtlasSpace_). `/CSHL_meshes/<atlas_name>/<atlas_name>_1um_meanPositions.pkl`

### Compute average shapes

- Mean shapes
    - `/CSHL_meshes/<atlas_name>/mean_shapes/<atlas_name>_10.0um_<sided_or_surround_structure>_meanShape_volume.bp`
    - `/CSHL_meshes/<atlas_name>/mean_shapes/<atlas_name>_10.0um_<sided_or_surround_structure>_meanShape_origin_wrt_meanShapeCentroid.txt`.
    
- Individual instances
  - Instance number to brain/side map. `/CSHL_meshes/<atlas_name>/instance_sources/<atlas_name>_<unsided_structure>_sources.pkl`
  - Instance-to-instance registration parameters
    - `/CSHL_meshes/<atlas_name>/mean_shapes/instance_registration/<unsided_structure>_instance<instance_num>/<registration_spec>/`.
      - `<registration_spec>_parameters.json`
      - `<registration_spec>_scoreEvolution.png`
      - `<registration_spec>_scoreHistory.bp`
      - `<registration_spec>_trajectory.bp`
    
### Put average shape at average position

- Atlas structures (located in _canonicalAtlasSpace_). 
    - `/CSHL_volumes/<atlas_name>/<atlas_name>_10.0um_scoreVolume/score_volumes/<atlas_name>_10.0um_scoreVolume_<sided_or_surround_structure>.bp`.
    - `/CSHL_volumes/<atlas_name>/<atlas_name>_10.0um_scoreVolume/score_volumes/<atlas_name>_10.0um_scoreVolume_<sided_or_surround_structure>_origin_wrt_canonicalAtlasSpace.txt`.


