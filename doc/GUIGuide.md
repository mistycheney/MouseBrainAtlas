# Create annotations using the graphical interface

There are two ways to place annotation on images.

## Method 1: 3-D reconstruction from manually drawn contours.

Create or edit contours. For complete instructions on how to interact with GUI, see [User Interface README](gui/README.md).

### Saving 2-D contours

Click "Save contours". All contours currently loaded are saved into the file `ANNOTATION_DIR/<stack>/<stack>_annotation_contours_<timestamp>.hdf`.

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

### Saving 3-D structures

Results are at `<stack>_annotation_structuresHanddrawn_<timestamp>.hdf`

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

## Method 2: Manual adjustment of pre-built atlas structures.




## Visualize and revise annotations using the labeling GUI

Download warped atlas maps into `VOLUME_DIR/<atlasName>/<atlasName>_<warp>_<fixedMapName>`.

Click "Load warped structures". Select structure. The structure contour (p=0.5) will be displayed over the images. You can move or rotate with respect to structure center. Each structure is manipulated as one integral 3-D entity. For complete instructions on how to interact with GUI, see [User Interface README](../gui/README.md).

Click "Save prob. structures". All structures currently loaded are saved into the file `ANNOTATION_DIR/<stack>/<stack>_annotation_probStructures_<timestamp>.hdf`.

Each row of the structure annotation file is indexed by a random `structure_id`. The columns are

* `volume_in_bbox`: bloscpack-encoded string of 3-D volume, isotropic voxel size ~ 0.45 micron * 32 = 14.4 micron.
* `bbox`: (xmin,xmax,ymin,ymax,zmin,zmax), wrt "wholebrain", same voxel size as above.
* `name`: structure name, unsided
* `side`: L or R or S
* `edits`. list. Each entry is a dict.
	* `username`
	* `timestamp`
	* `type`: shift3d or rotate3d
	* `transform`: flattened 3x4 matrix
	* `centroid_m`
	* `centroid_f`
