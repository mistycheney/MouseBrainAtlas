# Labeling GUI

To launch the GUI for thalamus labeling, run:
`~/Brain/gui/brain_labeling_gui_thalamus.py [stack_name] -f [first_section] -l [last_section]`

The left panel shows the full resolution images. The panels at the right are virtual coronal, horizontal and sagittal whole-brain slices, from top to bottom respectively. 

Left click and drag to pan. Scroll mouse wheels to zoom.

**Create new polygon**: Right click -> Select "New polygon" -> left click to place new vertices; press Enter to close a polygon.

**Set side**: Right click anywhere inside the polygon -> Select "Set side".

**Insert vertex**: Right click anywhere inside the polygon -> Select "Insert vertex" -> left click at anywhere between two existing vertices, and a new vertex will be placed there. -> Keep clicking to insert more vertices. Press ESC to finish.

**Delete vertex**: Right click anywhere inside the polygon -> Select "Delete vertex" -> left click and drag a box and all vertices inside the box will be deleted. -> Keep making boxes to delete more vertices. Press ESC to finish.

## Key bindings
- 1/2: go backward/forward along the stack in the full resolution panel.
- 3/4, 5/6, 7/8: go forward/backward in the coronal, horizontal, sagittal whole-brain overview panel, respectively.
- v: toggle whether vertices are shown.
- c: toggle whether contours are shown.
- l: toggle whether label are shown.
- q: switch to 3d shift mode.
- w: switch to 3d rotate mode.
- s: switch to show score map (need to have active polygon) / histology image
