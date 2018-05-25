This page describes the common data types in this system and how they are represented and stored.

Images
==========

A processed image has extension either `.tif` or `.jpg`. Data type is either unit8 or uint16.

Volumes
=======

A volume is represented by:
- a 3-D array stored as `bp` file ([bloscpack](https://github.com/Blosc/bloscpack)). Naming convention is `<volume_spec>.bp`.
- a (3,) int array representing the origin of this array with respect to _wholebrain_ (see [Definition of frames]), stored as `txt` file. Naming convention is `<volume_spec>_origin_wrt_wholebrain.txt`.

The following _volume types_ are defined, each with a different 3-d array data type:
- `annotationAsScore`: float, binary either 0 or 1
- `score`: float between 0 and 1
- `intensity`: uint8


Registration results
===========

For each registration, the following results are stored:
- `<registration_identifier>_parameters.json`: contains three keys `centroid_f_wrt_wholebrain`((3,)-array), `centroid_m_wrt_wholebrain`((3,)-array) and `parameters`((12,)-array).
- `<registration_identifier>_scoreHistory.bp`: the score history as a list
- `<registration_identifier>_scoreEvolution.png`: plot of the score over iterations
- `<registration_identifier>_trajectory.bp`: trajectory of the parameters during optimization, a list of 12 parameters.

Mathematically, a transform is expressed as:

q - q0 = R * T0(p-p0) + t

- `p` is a point in moving brain (wrt wholebrain in the case of a subject brain, or canonicalAtlasSpace in the case of atlas)
- `p0` is the rotation center defined on moving brain.
- `q` is a point in fixed brain (wrt wholebrain)
- `q0` is the shift of fixed brain.
- `R`, the 3x3 rotation matrix, which is part of the computed transform.
- `t`, 3-array, which is part of the computed transform.
- `T0` is the initial transform. 

In our system, a transform can be expressed in any of the following ways:

* dictionary
  - `parameters`: 12-array, flattened version of the rigid or affine 3x4 matrix.
  - `centroid_m_wrt_wholebrain`: 3-array, initial shift of the moving volume, relative to the wholebrain origin.
  - `centroid_f_wrt_wholebrain`: 3-array, initial shift of the fixed volume, relative to the wholebrain origin.
* (4,4) matrix: the 4x4 matrix that represents the transform.
* (3,4) matrix: first three rows of the full 4x4 matrix.
* (12,) array: flattened array of the first three rows of the full 4x4 matrix.


Annotation
=========

There are two types of annotations: 2-D polygons and 3-D structures. They are stored as HDF tabular files. Each row represents a polygon or a structure.

## 2-D Polygons 

Each row of the contour annotation file is indexed by a random `contour_id`. The columns are

* `class`: "contour" or "neuron"
* `creator`: username of the creator
* `downsample`: the downsample factor the vertices are defined on
* `edits`: the list of edits made on this contour
* `id`: a random uuid for this contour
* `label_position`: the position of the structure name text item relative to the whole image
* `name`: unsided name of this structure
* `orientation`: sagittal, coronal or horizontal
* `parent_structure`: currently not used
* `section`: the section number
* `side`: L or R
* `side_manually_assigned`: True if the side is confirmed by human; False if the side is automatically inferred.
* `time_created`: the time that the contour is created
* `type`: "intersected" if this contour is the result of interpolation or "confirmed" if confirmed by human
* `vertices`: vertices of a polygon. (n,2)-ndarray. wrt "prep2" crop, in unit of pixel at full resolution (~0.45 microns).
* `filename`: the file name of the section.

## 3-D Structures

Each row of this file contains information for one 3-D structure.
The columns are:
- `edits`: list of edit operations. Each operation is represented by a dict with the following keys:
  - `timestamp`: the time this edit is made
  - `transform`: 3x4 matrix representing the transform
  - `type`: one of _global_rotate3d_, _prob_shift3d_, _prob_rotate3d_
  - `username`: user who made this edit
- `name`: name of the structure
- `origin`: 3-D origin of the volume with respect to wholebrain frame, in unit of voxels.
- `resolution`: a string representing voxel size
- `side`: L or R or S (singular)
- `volume`: the 3-D volume encoded by bloscpack as string
