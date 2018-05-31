# User Guides
 Explanations for how to install and run the code, intended for the non-programmer, are in the [User Guide](doc/UserGuide.md).

# Code Structure

### Main Pipeline

- *annotation*: code related to processing human annotations
- *gui*: code for two GUIs 
  - brain labeling GUI (used to annotate landmarks in brain)
  - preprocessing GUI: processing raw images, remoing background, aligning stack.
- *learning*: code for extracting patches, generating features, training classifiers and processing score maps
- *preprocess*: code for preprocessing
- *reconstruct*: code for reconstructing volumes
- *registration*: code for 3D registration
- *utilities*: utilities code
- *visualization*: notebook for testing visualization code
- *web_service*: code for the web service that accept request for preprocessing and other tasks from GUI.
- *intensity*: codes for running intensity-based registration methods (via third-party program `elastix`).

### Utilities
- *utilities*: utility modules
- *3d*: code for rendering and displaying 3D models. Implementation uses VTK.
- *aws*: setup scripts for `cfncluster`.

### Side Projects
- *spm*: experiments about represent textures using SPM (Spatial Pyramid Matching)
- *snake*: experiments about active contour methods
- *adaboost_m2*
- *cells*: classification using cell-based texture descriptors.
- *dictionary*:	experiments for dictionary learning methods.
- *new_region*:
