


# Volume

A volume is represented by:
- a 3-D array stored as `bp` file.
- a (3,) int array representing the origin of this array with respect to _wholebrain_ (see [Definition of frames]), stored as `txt` file.

## Volume type
Three volume types are defined, each with a different 3-d array data type:
- `annotationAsScore`: float, binary either 0 or 1
- `score`: float between 0 and 1
- `intensity`: uint8

# Atlas

- Atlas.
  - Mean positions (wrt _canonicalAtlasSpace_). `/CSHL_meshes/<atlas_name>/<atlas_name>_1um_meanPositions.pkl`
  - Mean shape volumes
    - `/CSHL_meshes/<atlas_name>/mean_shapes/<atlas_name>_10.0um_<sided_or_surround_structure>_meanShape_volume.bp`
    - `/CSHL_meshes/<atlas_name>/mean_shapes/<atlas_name>_10.0um_<sided_or_surround_structure>_meanShape_origin_wrt_meanShapeCentroid.txt`.
  - Instance number to brain/side map. `/CSHL_meshes/<atlas_name>/instance_sources/<atlas_name>_<unsided_structure>_sources.pkl`
  - Instance-to-instance registration parameters
    - `/CSHL_meshes/<atlas_name>/mean_shapes/instance_registration/<unsided_structure>_instance<instance_num>/
/<registration_spec>/`.
      - `<registration_spec>_parameters.json`
      - `<registration_spec>_scoreEvolution.png`
      - `<registration_spec>_scoreHistory.bp`
      - `<registration_spec>_trajectory.bp`
    
