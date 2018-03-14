# Software installation

`$ pip install activeatlas`

This will download the scripts and the package containing the reference anatomical model and the trained texture classifiers.

Edit `global_setting.py`. In particular, specify the following variables:
- `DATA_DIR`
- `REPO_DIR`


# Preprocessing a new stack 

In this part, the user examines the images, creates tissue masks, aligns the sections in a stack and (optionally) crops the images.

## Data preparation

The raw images must be of sagittal sections, with the anterior at the left and the posterior at the right.

All images must be 16- or 8-bit tiff. To convert Zeiss scanner output from czi format to 16-bit tiff, use `CZItoTIFFConverter` from University of Geneva.

### Image Name ###

Each physical section is associated with an `imageName`.
There is no fixed composition rule for image names.
The principle is that one can trace back from an image name to the physical section. Therefore in each image name, these two elements are mandatory:
- slide number
- section or scene index in the slide

Other than that, the brain id is optional but desired. Other information such as the scan date or stain name are optional.
For example, both `CHATM3_slide31_2018_02_17-S2` and `Slide31-Nissl-S2` are valid image names.
It is important to use only one composition rule for each brain.

## Initialization

Create a JSON file that describes the image file paths. The file contains the following keys:
- `raw_data_dirs`
- `input_image_filename_mapping`
- `input_image_filename_to_imagename_re_pattern_mapping`
- `condition_list_fp`
- `ordering_rule_fp`

An example file is `CHATM3_input_specification.json`.

Run `$ initialize.py <input_spec_filepath>`.

Make sure the following items are generated under `DATA_DIR/<stack>`:
- `<stack>_sorted_filenames.txt`. In this file, each line contains an image name (without space), followed by its index in the series. The index is used to determine the z-position of a section, so any discarded section still occupies an index slot. You can ignore these sections or use the word "Placeholder" in place of their image names.
- `<stack>_thumbnail_Ntb`. This contains images named `<imageName>_thumbnail_Ntb.tif`. These can be either symbolic links or actual files.
- `<stack>_raw_Ntb`. This contains images named `<imageName>_raw_Ntb.tif`. These can be either symbolic links or actual files.
- `<stack>_raw_CHAT`. This contains images named `<imageName>_raw_Ntb.tif`. These can be either symbolic links or actual files.

## Intensity normalize fluorescent images

`$ normalize_intensity.py <stack> [input_version] [output_version]`

`<input_version>` default to "Ntb".
`<output_version>` default to "NtbNormalized"

Make sure the following items are generated under `DATA_DIR/<stack>`:
- `<stack>_thumbnail_NtbNormalized`

## Compute intra-stack transforms

`$ align.py <stack>`

This script computes a rigid transform between every pair of adjacent sections using the third-party program `elastix`.
On the workstation, with 8 processes, this takes about 1500 seconds.

Make sure the following items are generated under `DATA_DIR/<stack>`:
- `<stack>_elastix_output`.
- `<stack>_prep1_thumbnail_NtbNormalized`

## Launch the preprocessing GUI.

`$ gui/preprocess_tool_v3.py <stack>`

- Click "Load image order"
- Adjust the order by moving particular images up or down the stack.
- Click "Save image order"
- If the alignment between some pair requires correction, 
	- Click "Edit transforms"
	- Browse to the pair whose alignment needs correction.
	- Compute new alignment by supplying alternative parameter files to `elastix` or manually providing matching landmark points.

If any correction is made, make sure the following items are generated under `DATA_DIR/<stack>`:
- `<stack>_custom_transforms`

## Launch the preprocessing GUI.

`$ gui/preprocess_gui.py <stack>`

1. The GUI displays the thumbnails, sorted according to a hard-coded `imageName`-based sorting rule. You can adjust the order by changing the image assigned to each slice slot.
2. Click "Align" button. The GUI invokes `elastix` to compute a transform between each consecutive section pair. The automatically computed transform parameters are stored in `<stack>_elastix_output`. Once finished, you can review the pairwise transforms for every pair. You can correct the wrong ones by manually placing anchor points and request a re-compute. These manually specified transforms are stored in `<stack>_custom_transforms`.
3. Once all pairwise transforms are reviewed, select a target section that you want all sections to register towards. The default is the largest section. The imageName of the target section is stored in `<stack>_anchor.txt`.
4. Click "Compose" button. The program composes all pairwise transforms and aligns every section to the target section. Aligned images are generated in the folder `<stack>_prep1_thumbnail`. Once finished, the aligned stack is loaded in the "Aligned" panel.
5. Browse and inspect the aligned stack. If there are ordering errors or alignment errors between particular pairs, repeat step 1-4.
6. Draw a 2-D crop box on top of the aligned stack, that contains only the brain region of interest.
7. Set the first section and the last section that contains only the brain region of interest. These together with the 2-D crop box define a 3-D ROI for the reconstructed specimen volume. The cropbox coordinates and the two section limits are stored in file `<stack>_alignedTo_<anchorImageName>_cropbox.txt`.
8. Click "Crop" button. The GUI invokes ImageMagick `convert` to crop the thumbnail images, transform and crop the raw images. The cropped thumbnail images are stored in `<stack>_prep2_thumbnail`. The cropped raw images are stored in `<stack>_prep2_lossless`. Note that because the transform and crop of the raw images is done in one-shot, we do not store the aligned pre-crop raw images.

Step 3: Launch the mask editing GUI.

`$ mask_editing_tool_v3.py <stack>`

1. Upon starting, the GUI shows the aligned uncropped thumbnail image stack. The program uses snake to evolve contours and requires an initial contour that completely encircles the tissue to be provided for every section. You are only required to draw such initial contours on 5-10 sections where the extent of the tissue changes abruptly. On such a section, click "Create initial contour", and then place vertices consecutively to create a contour.
2. Once done drawing initial contours on all selected sections, click "Interpolate". The contours for the remaining sections are automatically generated by interpolation.
3. Click "Shrink". The program evolves the initial contour on every section towards the tissue (using python package `morphsnake`). For many sections, this procedure produces multiple disjoint submasks. Once finished, the GUI shows on top of the thumbnail images the contour for each submask. All submasks are stored in `<stack>_submasks`. It is often hard for the program to judge whether a submask is normal tissue or dirt/debris. Therefore, before exporting the final mask, we add a human verification step in which a human verifies the validity of each submask. 
4. Click on each submask contour to toggle whether the submask is valid or not. If the auto-generated submasks are not accurate, you can also create new submasks by drawing on the images. Submasks of all sections that are modified are stored in `<stack>_submasks_modified`. Once done, click "Save masks". The final masks as binary images are generated in `<stack>_prep1_thumbnail_mask`.
5. Click "Crop" button to generate cropped versions of the masks. They are stored in `<stack>_prep2_thumbnail_mask`.


# Processing a new stack given a trained atlas

`learning/pipeline_aws.ipynb`

Step 1: Convert image to scoremap.

`$ ./images_to_scoremaps.py <stack>`

Step 2: Reconstruct score volume and compute spatial gradient.

`$ ./reconstruct_score_volume.py <stack>`

Step 3: Register to atlas.

`$ ./register.py <transform_spec>`

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


# Training the atlas: 

## Create annotations using the labeling GUI

If you want to edit previously saved contours, click "Load contours".

Create or edit contours. For complete instructions on how to interact with GUI, see [User Interface README](../gui/README.md).

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

