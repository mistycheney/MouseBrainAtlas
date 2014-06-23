BrainSaliencyDetection
======================

Framework for detecting salient regions in mouse brain images

Project files are at `/oasis/projects/nsf/csd181/yuncong/Brain`

Data are at `/oasis/projects/nsf/csd181/yuncong/ParthaData`

Output are at `/oasis/scratch/csd181/yuncong/output`

Usage
-----

The main script is an ipython notebook `ManagerScriptV1` under `notebooks/`.

In the notebook, specify the inputs:
* dataset name, e.g. `dataset_name = 'PMD1305_reduce0_region0'`
* image index, e.g. `image_idx = 244`
* parameter id, e.g. `param_id = 5`

The notebook then calls the script `CrossValidationPipelineScriptShellNoMagicV1.py` with proper arguments.

<a name="param"></a> Parameters
-----
Parameters are set [in this spreadsheet](https://docs.google.com/spreadsheets/d/1S189da_CxzC3GKISG3hZDG0n7mMycC0v4zTiRJraEUE/edit#gid=0)

* **param_id**: an integer id for this set of parameters, default parameter is id 0
* **min_wavelen**: minimum wavelength of Gabor filter, in number of pixels, default 5
* **max_wavelen**: maximum wavelength of Gabor filter, in number of pixels, default 40
* **freq_step**: multiply factor to the next frequency, default 2
* **theta_interval**: interval of Gabor filter orientations, in degrees, defauly 15.
* **n_superpixels**: desired number of superpixels in the over-segmentation, default 100. Large number leads to smaller superpixels. Default 100.
* **slic_compactness**: larger value leads to more square segmentation. Default 5.
* **slic_sigma**: width of Gaussian kernel used in pre-smoothing before segmentation. Default 10.
* **n_texton**: number of texton, or the number of clusters when doing rotation-invariant k-means over Gabor responses.



ManagerScriptV1 sets and calls CrossValidationPipelineScriptShellNoMagicV1.py
