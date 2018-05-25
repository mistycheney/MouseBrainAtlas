# Naming convention for processed images #

Every processed image can be uniquely identified by the following information:
- image name: See below.
- version: a word that specifies a channel or the processed result of a particular channel. For example, it can be "Ntb" for neurotrace channel, "CHAT" for the ChAT channel, "NtbNormalized" for the intensity-normalized neurotrace channel.
- resolution: a word that specifies the pixel resolution. It can be "raw", "down32" (downsample raw data in both dimensions by a factor of 32), "thumbnail" (same as "down32"). It can also be an absolute physical size such as "10.0um".
- prep id: a number or word that identifies the preprocessing procedure applied, such as rotation and cropping.

Processed images are stored under `$DATA_ROOTDIR`. The path to each processed image follows the pattern `<stack>/<stack>_prep<prep_id>_<resol>_<version>/<image_name>_prep<prep_id>_<resol>_<version>.<ext>`.

### Image Name ###

Each physical section is associated with an `imageName`.
There is no fixed composition rule for image names.
The principle is that one can trace back from an image name to the physical section. Therefore in each image name, these two elements are mandatory:
- slide number
- section or scene index in the slide

Other than that, the brain id is optional but desired. Other information such as the scan date or stain name are optional.
For example, both `CHATM3_slide31_2018_02_17-S2` and `Slide31-Nissl-S2` are valid image names.
It is important to use only one composition rule for each brain. **Do not use space or special characters such as ampersand** as they will not be parsed correctly in Linux commandline.

### Preprocessing Identifier (prep id) ###
- None: original unaligned uncropped image
- 1: alignedPadded
- 2: alignedBrainstemCrop
- 5: alignedWithMargin

# General Steps:

## Convert raw data to TIFs
- jp2 -> raw

## For thionin (brightfield) data
* raw -> thumbnail
* **Compute tranforms using thumbnail**
* thumbnail -> prep1_thumbnail
* **Supply prep1_thumbnail_mask**
* prep1_thumbnail_mask -> thumbnail_mask
* raw -> prep1_raw
* **Compute prep5 (alignedWithMargin) cropping box based on prep1_thumbnail_mask**
* prep1_raw -> prep5_raw:
* prep1_thumbnail -> prep5_thumbnail:
* prep1_thumbnail_mask -> prep5_thumbnail_mask:
* **Specify prep2 (alignedBrainstemCrop) cropping box**
* prep5_raw -> prep2_raw
* prep5_thumbnail -> prep2_thumbnail
* prep5_thumbnail_mask -> prep2_thumbnail_mask
* prep2_raw -> prep2_raw_gray: 30s * ~300 sections = 150 mins
* prep2_raw_gray -> prep2_raw_grayJpeg: 20s * ~300 sections = 100 mins
* prep2_raw -> prep2_raw_jpeg: 20s * ~300 sections = 100 mins

_prep2_raw_gray_ are used for structure detection.
_prep5_raw_ will be published online.

## For Neurotrace (fluorescent) data
* raw -> raw_Ntb:
* raw_Ntb -> thumbnail_Ntb
* thumbnail_Ntb -> thumbnail_NtbNormalized: 0.1s/section
* **Compute transforms using thumbnail_NtbNormalized**
* **Supply prep1_thumbnail_mask**
* prep1_thumbnail_mask -> thumbnail_mask
* raw_Ntb -> raw_NtbNormalizedAdaptiveInvertedGamma (**brightness correction**)
* **Compute prep5 (alignedWithMargin) cropping box based on prep1_thumbnail_mask**
* raw_NtbNormalizedAdaptiveInvertedGamma -> prep5_raw_NtbNormalizedAdaptiveInvertedGamma: ~1.5min/section * 300 sections = 7.5 hrs
* thumbnail_NtbNormalized -> prep5_thumbnail_NtbNormalized: 70s/stack (8 threads)
* prep5_raw_NtbNormalizedAdaptiveInvertedGamma -> prep5_thumbnail_NtbNormalizedAdaptiveInvertedGamma: 5s/section
* **Specify prep2 (alignedBrainstemCrop) cropping box**
* prep5_raw_NtbNormalizedAdaptiveInvertedGamma -> prep2_raw_NtbNormalizedAdaptiveInvertedGamma: 1500s/stack (4 threads)
* prep2_raw_NtbNormalizedAdaptiveInvertedGamma -> prep2_raw_NtbNormalizedAdaptiveInvertedGammaJpeg: 14s/section


# jp2/czi -> raw

## CSHL data
Data from CSHL are acquired using Hamamatsu Nanozoomer (0.46 micron/pixel).
Raw data from the scanner are NDPI files. 
The raw files are of whole-slides and do not specify the bounding box of individual sections.
CSHL did the segmentation and sent us images of individual sections re-encoded as JPEG2000 files.
(Note: we do not have the segmentation code at this moment.) 

### Convert JPEG2000 to TIF
Use [Kakadu](http://kakadusoftware.com/downloads/). Run `export LD_LIBRARY_PATH=<kdu_dir>:$LD_LIBRARY_PATH; <kdu_bin> -i <in_fp> -o <out_fp>`.

Output are 8-bit (thionin) or 16-bit (fluorescent) TIFFs.

## UCSD data
UCSD data are acquired using Zeiss Axioscan (0.325 micron/pixel).
Raw data from the scanner are CZI files. In these files individual sections are recorded as different scenes.

### Convert CZI to TIFF
Use [CZItoTIFFConverter](http://cifweb.unil.ch/index.php?option=com_content&task=view&id=152&Itemid=2) ([user manual](https://www.unige.ch/medecine/bioimaging/files/7814/3714/1634/CZItoTIFFConverter.pdf)).

Use the graphical interface with the following settings:
- Create BigTIFF files (check)
- One file per scene (check)
- Use JPEG (check)
- Enforce LZW for FL (check)
- Use DisplaySetting for FL images (uncheck)
- Use Channel Names (check)
- Use Scene Names (check)

Output are 8-bit (thionin) or 16-bit (fluorescent) TIFFs.

# raw -> Ntb/CHAT or raw -> gray

Extract from data a single channel that shows Nissl cytoarchitecture and optionally, another single channel that shows cell markers.

For Nanozoomer fluorescent images (e.g. those from CSHL), use the blue channel for Neurotrace, green or red channel for markers.

For Axioscan fluorescent images, channels are labeled with meaningful names.

For Nissl images, convert RGB to grayscale.


# Rectify images

The images must have anterior at the left, posterior at the right, dorsal at the top and ventral at the bottom.

# Preprocessing a new stack

In this part, the user examines the images, creates tissue masks, aligns the sections in a stack and (optionally) crops the images.

## Initialization

Create a JSON file that describes the image file paths. The file contains the following information:
- `raw_data_dirs`: this is a dict with a list of (version, resolution)-tuples as keys. For example they keys might be ('CHAT', 'raw') or ('Ntb', 'thumbnail').
- `input_image_filename_to_imagename_re_pattern_mapping`

Example json file content:

```
raw_data_dirs = \
{('Ntb', 'raw'): '/media/yuncong/BethandHannah_External1/CHATM2/CHATM2_raw_Ntb',
('CHAT', 'raw'): '/media/yuncong/BethandHannah_External1/CHATM2/CHATM2_raw_CHAT',
('AF', 'raw'): '/media/yuncong/BethandHannah_External1/CHATM2/CHATM2_raw_AF',
('Ntb', 'thumbnail'): None}

input_image_filename_to_imagename_re_pattern_mapping = \
{('Ntb', 'raw'): \
 '/media/yuncong/BethandHannah_External1/CHATM2/CHATM2_raw_Ntb/(.*)_.*?_.*?.tif',
 ('CHAT', 'raw'): \
 '/media/yuncong/BethandHannah_External1/CHATM2/CHATM2_raw_CHAT/(.*)_.*?_.*?.tif', 
 ('AF', 'raw'): \
 '/media/yuncong/BethandHannah_External1/CHATM2/CHATM2_raw_AF/(.*)_.*?_.*?.tif', 
}
```

The image file path mappings are indexed by (version, resolution) tuples.

An example file is `CHATM3_input_specification.json`.

Run `$ initialize.py <input_spec_filepath>`.

Make sure the following items are generated under `DATA_DIR/<stack>`:
- `<stack>_sorted_filenames.txt`. In this file, each line contains an image name (without space), followed by its index in the series. The index is used to determine the z-position of a section, so any unused section still occupies an index slot. For these sections, use the word "Placeholder" in place of image name.
- `<stack>_thumbnail_Ntb`. This contains images named `<imageName>_thumbnail_Ntb.tif`. These can be either symbolic links or actual files.
- `<stack>_raw_Ntb`. This contains images named `<imageName>_raw_Ntb.tif`. These can be either symbolic links or actual files.
- `<stack>_raw_CHAT`. This contains images named `<imageName>_raw_CHAT.tif`. These can be either symbolic links or actual files.

## Intensity normalize fluorescent images

`$ normalize_intensity.py <stack> [input_version] [output_version]`

`<input_version>` default to "Ntb".
`<output_version>` default to "NtbNormalized"

Make sure the following items are generated under `DATA_DIR/<stack>`:
- `<stack>_thumbnail_NtbNormalized`

## Compute intra-stack transforms

`$ align.py <stack>`

This script computes a rigid transform between every pair of adjacent sections using the third-party program `elastix`.
It then selects an anchor section (by default this is the largest section in the stack) and concatenate the adjacent transforms to align every section to match the anchor.

On the workstation, with 8 processes, this takes about 30 minutes.

Make sure the following items are generated under `DATA_DIR/<stack>`:
- `<stack>_elastix_output`
- `<stack>_prep1_thumbnail_NtbNormalized`
- `<stack>_anchor.txt`

## Review alignment

`$ gui/preprocess_tool_v3.py <stack>`

- Click "Load image order"
- Adjust the order by moving particular images up or down the stack.
- Click "Save image order"
- If the alignment between any pair requires correction,
	- Click "Edit transforms"
	- Browse to the pair whose alignment needs correction.
	- Compute new alignment either by re-running `elastix` using another parameter file or manually providing matching landmark points.

If any correction is made, make sure the following items are generated under `DATA_DIR/<stack>`:
- `<stack>_custom_transforms`

Then re-run
`$ align.py <stack>`

## Specify cropping

Cropping of the images is desired to focus computation only on the region of interest.

Select "Show cropbox". Draw a 2-D crop box.
Also set the first section and the last section.

Make sure the following items are generated under `DATA_DIR/<stack>`:
- `<stack>_alignedTo_<anchorImageName>_cropbox.txt`. This file contains one line (xmin, xmax, ymin, ymax, sec_min, sec_max).

Run
`$ crop.py <stack>`

This invokes ImageMagick `convert` to crop the thumbnail images and transform and crop the raw images. Note that because the transform and crop of the raw images is done in one command, we do not store the aligned but non-cropped raw images.

Make sure the following items are generated under `DATA_DIR/<stack>`:
- `<stack>_prep2_thumbnail_<channel>`
- `<stack>_prep2_raw_<channel>`

## Generate masks

`$ mask_editing_tool_v3.py <stack>`

1. Upon starting, the GUI shows the aligned uncropped thumbnail image stack. The program uses snake to evolve contours and requires an initial contour that completely encircles the tissue to be provided for every section. The contours should be as tight as possible because the time for contour evolution will be minimized.
You are only required to draw such initial contours on 5-10 sections where the extent of the tissue changes abruptly. On such a section, click "Create initial contour", and then place vertices consecutively to create a contour.
2. Once done drawing initial contours on all selected sections, click "Interpolate". The contours for the remaining sections are automatically generated by interpolation.
3. Click "Shrink". The program evolves the initial contour on every section towards the tissue (using python package `morphsnake`). For many sections, this procedure produces multiple disjoint submasks. Once finished, the GUI shows on top of the thumbnail images the contour for each submask. All submasks are stored in `<stack>_submasks`. It is often hard for the program to judge whether a submask is normal tissue or dirt/debris. Therefore, before exporting the final mask, we add a human verification step in which a human verifies the validity of each submask.
4. Click on each submask contour to toggle whether the submask is valid or not. If the auto-generated submasks are not accurate, you can also create new submasks by drawing on the images. Submasks of all sections that are modified are stored in `<stack>_submasks_modified`. Once done, click "Save masks". The final masks as binary images are generated in `<stack>_prep1_thumbnail_mask`.
5. Click "Crop" button to generate cropped versions of the masks. They are stored in `<stack>_prep2_thumbnail_mask`.


## Generate 3-D intensity volume #

Run `reconstruct/construct_intensity_volume.py [--tb_version <tb_version>] [--tb_resol <tb_resol>] [--output_resol <output_resol>]`

Output is at `/CSHL_volumes/<brain_name>/<brain_name>_wholebrainWithMargin_10.0um_intensityVolume/`.

Notebook: `reconstruct/construct_intensity_volume_v3.ipynb`
