# Create annotations using the labeling GUI

There are two modes to place annotation on images.

## 3-D reconstruction from manually drawn contours.

## Manual adjustment of pre-built atlas structures.

If you want to edit previously saved contours, click "Load contours".

Create or edit contours. For complete instructions on how to interact with GUI, see [User Interface README](gui/README.md).

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
