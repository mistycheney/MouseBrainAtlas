


# Volume

A volume is represented by:
- a 3-D array stored as `bp` file.
- a (3,) int array representing the origin of this array with respect to _wholebrain_ (see [Definition of frames]), stored as `txt` file.

## Volume type
Three volume types are defined, each with a different 3-d array data type:
- `annotationAsScore`: float, binary either 0 or 1
- `score`: float between 0 and 1
- `intensity`: uint8


# Mesh

Visualize meshes using [slicer](https://download.slicer.org/).

STL files are under:
- Subject.
`/home/yuncong/CSHL_meshes/<brain_name>/<brain_name>_10.0um_annotationAsScoreVolume/<brain_name>_10.0um_annotationAsScoreVolume_<sided_structure>.stl`
- Atlas.
  - Mean shape meshes (different levels). `/CSHL_meshes/<atlas_name>/mean_shapes/<atlas_name>_10.0um_<sided_or_surround_structure>_meanShape_mesh_level<level>.stl`
  - Mean shape volumes
    - `/CSHL_meshes/<atlas_name>/mean_shapes/<atlas_name>_10.0um_<sided_or_surround_structure>_meanShape_volume.bp`
    - `/CSHL_meshes/<atlas_name>/mean_shapes/<atlas_name>_10.0um_<sided_or_surround_structure>_meanShape_origin_wrt_meanShapeCentroid.txt`.
  - instance mesh. `/CSHL_meshes/<atlas_name>/aligned_instance_meshes/<atlas_name>_10um_<unsided_structure>_<instance_num>.stl`
  - instance number to brain/side map. `/CSHL_meshes/<atlas_name>/instance_sources/<atlas_name>_<unsided_structure>_sources.pkl`
  - Instance-to-instance registration parameters
    - `/CSHL_meshes/<atlas_name>/mean_shapes/instance_registration/<unsided_structure>_instance<instance_num>/
/<registration_spec>/`.
      - `<registration_spec>_parameters.json`
      - `<registration_spec>_scoreEvolution.png`
      - `<registration_spec>_scoreHistory.bp`
      - `<registration_spec>_trajectory.bp`
    
