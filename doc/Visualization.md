# Rendering structures as meshes.

3-D meshes are stored as STL files.
An STL file contains 
- 3-D coordinates of the vertices 
- a (n,3)-list of vertex indices; each row represents a triangular face.

STL files can be visualized using [slicer](https://download.slicer.org/).

STL files are under:
- Subject.
`/home/yuncong/CSHL_meshes/<brain_name>/<brain_name>_10.0um_annotationAsScoreVolume/<brain_name>_10.0um_annotationAsScoreVolume_<sided_structure>_l<level>.stl`

- Atlas.
  - Structure meshes (different levels, located in _canonicalAtlasSpace_).  `/CSHL_meshes/<atlas_name>/<atlas_name>_10.0um_scoreVolume/<atlasname>_10.0um_scoreVolume_<sided_or_surround_structure>.stl`
  - Mean positions (wrt _canonicalAtlasSpace_). `/CSHL_meshes/<atlas_name>/<atlas_name>_1um_meanPositions.pkl`
  - Mean shape meshes (different levels). `/CSHL_meshes/<atlas_name>/mean_shapes/<atlas_name>_10.0um_<sided_or_surround_structure>_meanShape_mesh_level<level>.stl`
  - Instance mesh. `/CSHL_meshes/<atlas_name>/aligned_instance_meshes/<atlas_name>_10um_<unsided_structure>_<instance_num>.stl`
  - Instance number to brain/side map. `/CSHL_meshes/<atlas_name>/instance_sources/<atlas_name>_<unsided_structure>_sources.pkl`

# Rendering 3-D probability map as translucent cloud

