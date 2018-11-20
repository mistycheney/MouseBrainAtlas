 For complete instructions on how to interact with GUI, see [User Interface README](../gui/README.md).

# Create/modify annotations using the graphical interface

There are two ways to manipulate annotation on images.

## Method 1: Manually draw contours.

Create or edit contours.

### Saving 2-D contours

Click "Save contours". All contours currently loaded are saved into the file `ANNOTATION_DIR/<stack>/<stack>_annotation_contours_<timestamp>.hdf`.

### Saving 3-D structures

Results are at `<stack>_annotation_structuresHanddrawn_<timestamp>.hdf`

Also see [explanation of annotation file](FileOrganization.md)

## Method 2: Manual adjustment of pre-built atlas structures.


# Check annotation content

Check annotation entries using `annotation/check_annotation_file_v3.ipynb`.

Generate annotation contour overlaid images using `annotation/visualize_annotations_v5.ipynb`.
