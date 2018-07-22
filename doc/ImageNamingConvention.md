# Naming convention for processed images #

Every processed image can be uniquely identified by a combination of the following information:
- **image name**: a string that uniquely identifies a physical section.
- **prep id**: a number or word that identifies the _spatial_ adjustment operations applied.
- **version**: a word that specifies particular channel or _appearance_ adjustment operations.
- **resolution**: a word that specifies the pixel resolution.

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
- 1 (alignedPadded): section-to-section aligned, with large paddings.
- 5 (alignedWithMargin): tightly crop over full tissue area with fixed small margin on all four sides.
- 2 (alignedBrainstemCrop): crop only the brainstem area (from caudal end of thalamus to caudal end of trigeminal caudalis, from top of superior colliculus to bottom of brain)

### Resolution ###
- raw: original resolution of images. (0.325 micron/pixel for Axioscan, 0.46 micron/pixel for Nanozoomer)
- thumbnail: usually 32x downscaled from raw
- 10.0um: 10 micron

### Version ###
- Ntb: Neurotrace blue channel. Single-channel image.
- NtbNormalized: linearly intensity stretched.
- NtbNormalizedAdaptiveInvertedGamma: locally adaptive intensity normalized.
- CHAT: ChAT signal channel. Single-channel image.
- gray: single-channel grayscale images converted from RGB thionin images using skimage's `rgb2gray`.
- mask: binary mask (1 for the tissue area, 0 for slide background).
