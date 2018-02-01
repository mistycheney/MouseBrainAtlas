# Labeling GUI

To launch the labeling GUI, run:

`~/Brain/gui/brain_labeling_gui_v27.py [stack_name] -v jpeg -p [prep_id]`

`prep_id` is 2 for brainstem crop and 3 for thalamus crop.

The left panel shows the full resolution images. The panels at the right are virtual coronal, horizontal and sagittal whole-brain slices, from top to bottom respectively.

Left click and drag to pan. Scroll mouse wheel to zoom.

**Create new polygon**: Right click -> Select "New polygon" -> left click to place new vertices; press Enter to close a polygon.

**Set side**: Right click anywhere inside the polygon -> Select "Set side".

**Insert vertex**: Right click anywhere inside the polygon -> Select "Insert vertex" -> left click at anywhere between two existing vertices, and a new vertex will be placed there. -> Keep clicking to insert more vertices. Press ESC to finish.

**Delete vertex**: Right click anywhere inside the polygon -> Select "Delete vertex" -> left click and drag a box and all vertices inside the box will be deleted. -> Keep making boxes to delete more vertices. Press ESC to finish.

**Save**: Click "Save contours" in the bottom row. The annotations on the full resolution panel will be saved into `~/CSHL_labelings_thalamus/[stack]/[stack]_annotation_contours_[timestamp].hdf`. Annotations for the coronal and horizontal panels are also saved. One can read and inspect the annotation files as shown in [this notebook](https://github.com/mistycheney/MouseBrainAtlas/blob/master/annotation/check_annotation_file_v3_for_thalamus.ipynb).

**Load**: Click "Load contours" in the bottom row -> Select the latest `[stack]_annotation_contours_[timestamp]` according to the modified time.

## Key bindings
- 1/2: go backward/forward along the stack in the full resolution panel.
- 3/4, 5/6, 7/8: go forward/backward in the coronal, horizontal, sagittal whole-brain overview panel, respectively.
- v: toggle whether vertices are shown.
- c: toggle whether contours are shown.
- l: toggle whether label are shown.
- hold ctrl: 2d shift
- hold alt: 2d rotate
- t: switch to prob 3d shift mode. Then move polygon.
- r: switch to prob 3d rotate mode. Then rotate the polygon. Rotation center is assumed to be the centroid of the 2d contour in current section.
- alt + t: switch to global 3d shift mode.
- alt + r: switch to global 3d rotation mode.
- s: switch to show score map (need to have active polygon) / histology image
