cd This is the repo for mouse brainstem atlas project.

- *3d*: code for rendering and displaying 3D models. Implementation uses VTK.
- *annotation*: code related to processing human annotations
- *cells*: experiments for cell-based texture representation
- *dictionary*: experiments for dictionary learning
- *gui*: code for two GUIs - brain labeling GUI and preprocessing GUI
- *learning*: code for extracting patches, generating features, training classifiers and processing score maps
- *preprocess*: code for preprocessing
- *reconstruct*: code for reconstructing volumes
- *registration*: code for 3D registration
- *snake*: experiments about active contour methods
- *spm*: experiments about represent textures using SPM (Spatial Pyramid Matching)
- *utilities*: utilities code
- *visualization*: notebook for testing visualization code
- *web_service*: code for the web service that accept request for preprocessing and other tasks from GUI.

# Possible Improvement #
- Use a unbiased way to construct/update reference model, rather than align all brains to one particular brain.

# Initial Training #
`learning/train_classifiers_v3.py`.

**Requires:**
- `$ANNOTATION_ROOTDIR/[stack]/[stack]_annotation_grid_indices.h5` for all annotated stacks.

# Pipeline for Unannotated Specimens #

## Preprocess ##
`preprocess/preprocess_pipeline_aws.ipynb`

## Feature Generation ##
`learning/extract_test_features_cnn.ipynb`
This uses MXNet.

Extract the first

vgg16-blue 500s / section (why so slow??)
inception-bn-blue 160s / section


Pre-trained Inception-BN network on RGB patches of 224 by 224 pixels.

Use lossless (contrast stretched 8-bit version for fluorescent images).
Patches are extracted at grid points with horizontal and vertical spacing of 56 pixels.
~100k patches per section.
For each section, the execution time is 6 minutes, which breaks down into:
- load image: 113.57 seconds
- extract patches: 42.50 seconds
- predict: 210.46 seconds



`learning/pipeline_aws.ipynb`
## Classification ##

- First we apply the classifiers to the images using `apply_classifiers_v3.py`. It is a multi-process code over sections. It generates scores on a sparse grid locations. Outputs are in `$SPARSE_SCORES_ROOTDIR`.

- Resample sparse scores.
Specify a resolution. Resample the score maps by the resolution.
Resampled score maps are used to generate score volumes and score map visualizations.

- Construct score volumes.
Specify a resolution. Load corresponding score maps. Stack the score maps up to form score volumes.
Script `construct_score_volume_v4.py` for one structure. Single-process program (?). Distribute structures over cluster.
Outputs are in `$VOLUME_ROOTDIR`. The volume as a 3D numpy array `volume` and the bounding box (xmin,xmax,ymin,ymax,zmin,zmax) `bbox`.
A lot of outputs involved, so it is better to use local /scratch.

- Visualize score maps (optional).
Specify a resolution. Load corresponding score maps. Generated visualizations are JPEG images at `$SCOREMAP_VIZ_ROOTDIR`. Heatmap is `plt.cm.hot`.
Script `visualize_scoremaps_v3.py`.

## Registration ##
- Global alignment.


# Memory Usage #
- Global registration: 32GB RAM is not enough.
Ideally, each score volume has ~500^3 = 125M voxels x (moving vol 2Bytes, moving grad 4Bytes, fixed vol 2Bytes, fixed grad 4Bytes) = 1.5GB. Then x 14 selected structures = 21GB.
10G free out of 64G.
Can only do one global registration on a node due to high RAM requirement.
- Transform: simultaneous `NUM_CORES` processes each stack, one for each structure.
- Visualize registration: simultaneous `NUM_CORES` processes each stack, one for each structure.
- Local registration:
