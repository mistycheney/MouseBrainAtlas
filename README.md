This is the repo for mouse brainstem atlas project.

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

`learning/train_classifiers_v3.py`

**Requires:**
- `$ANNOTATION_ROOTDIR/[stack]/[stack]_annotation_grid_indices.h5` for all annotated stacks.
