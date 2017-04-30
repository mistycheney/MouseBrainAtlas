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
`learning/pipeline_aws.ipynb`

- First we apply the classifiers to the images using `apply_classifiers_v3.py`. It is a multi-process code over sections. It generates scores on a sparse grid locations.

- Construct score volume.
Specify a resolution. Resample the scoremaps according to the resolution. Stack up to form score volume.



# Memory Usage #
- Global registration: 32GB RAM is not enough.
Ideally, each score volume has ~500^3 = 125M voxels x (moving vol 2Bytes, moving grad 4Bytes, fixed vol 2Bytes, fixed grad 4Bytes) = 1.5GB. Then x 14 selected structures = 21GB.
10G free out of 64G.
Can only do one global registration on a node due to high RAM requirement.
- Transform: simultaneous `NUM_CORES` processes each stack, one for each structure.
- Visualize registration: simultaneous `NUM_CORES` processes each stack, one for each structure.
- Local registration:
