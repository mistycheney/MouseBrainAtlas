BrainSaliencyDetection
======================

Framework for detecting salient regions in mouse brain images

Project files are at `/oasis/projects/nsf/csd181/yuncong/Brain`

Data are at `/oasis/projects/nsf/csd181/yuncong/ParthaData`

Output are at `/oasis/scratch/csd181/yuncong/output`

**Yoav** to check that these are public

Usage
-----

The executable script `CrossValidationPipelineScriptShellNoMagicV1.py` is under project directory `Brain/notebooks`.
Run `python CrossValidationPipelineScriptShellNoMagicV1.py -h` to see the usage.

Output will be generated in the sub-directory `<dataset name>_<image index>_param<parameter id>` under the output path. One can download the output by running [`download_all.sh`](https://gist.github.com/mistycheney/8e31ea126e23011871e6) on the local machine.
Just type `./download_all.sh` to see the usage.

Data
----
The original data are tar.gz files. Use [`preprocess.py`](https://gist.github.com/mistycheney/4e5cafdf049b9cdc478c) under the data directory to generate a new dataset.
<pre>
usage: preprocess.py [-h] [-b BBOX_FILE] [-i DATA_DIR] [-o OUT_DIR]
                     stack_name start_frame stop_frame reduce_level
</pre>
**Yuncong** add in the script more documentation, including the format of the bbox file etc.

Data from Partha:
Original data is in the directories whose names are PMD1305  PMD1328  PMD1337  PMD1340  PMD1342

Data from David: 

Files after processing are in directories with names of the form: PMD1305_reduce0_region0  where reduce0 means no reduction in resolution and region0 defines a bounding box containing the stem.

Output
-----
A result set is the set of outputs from a particular image using a particular parameter setting. All results are stored in the output directory. Each result set is a sub-directory named `<dataset name>_<image index>_param<parameter id>`. The content of each sub-directory are .npy files or image files with different `_<suffix>`. 

`<suffix>` is one of:
* `features.npy`: Gabor filter responses. shape (`im_height`, `im_width`, `n_features`)
* `segmentation.npy`: same size as the image, each pixel is an integer indicating the index of the suerpixel it belongs to.
* `sp_dir_hist_normalized.npy`: energy distribution over different directions for each superpixel, shape (`n_superpixel`, `n_angle`)
* `sp_texton_hist_normalized.npy`: texton distribution for each superpixel, shape (`n_superpixel`, `n_texton`)
* `textonmap.npy`: same size as the image, each pixel is an integer from 0 to `n_texton-1` or -1 for background, shape (`im_height`, `im_width`)

* `segmentation.png`
* `textonmap.png`
* `texton_saliencymap.png`


Running `ssh gcn 'ls -d Brain/output/*/'` from local machine returns a list of available results.


<a name="param"></a> Parameters
-----
Parameters are specified in [this spreadsheet](https://docs.google.com/spreadsheets/d/1S189da_CxzC3GKISG3hZDG0n7mMycC0v4zTiRJraEUE/edit#gid=0). ManagerScriptV1 will automatically read this spreadsheet and know about all the parameter settings. If you have made some change to the spreadsheet, set the flag `redownload=True` for the `load_parameters` function in ManagerScriptV1.


#### Gabor filter bank parameters ##
* **param_id**: an integer id for this set of parameters, default parameter is id 0
* **min_wavelen**: minimum wavelength of Gabor filter, in number of pixels, default 5
* **max_wavelen**: maximum wavelength of Gabor filter, in number of pixels, default 40
* **freq_step**: multiply factor to the next frequency, default 2 (same as factor for wavelength)
* **theta_interval**: interval of Gabor filter orientations, in degrees, defauly 15.

**Yuncong** What about the other parameters that define a filter: the width of the gaussian envelope, the size of the filter in pixels. I would like those exposed, and their default value equal to the current setting. Are there any other parameters?


#### Super-Pixel parameters ##
* **n_superpixels**: desired number of superpixels in the over-segmentation, default 100. Large number leads to smaller superpixels. Default 100.
* **slic_compactness**: larger value leads to more square segmentation. Default 5.
* **slic_sigma**: width of Gaussian kernel used in pre-smoothing before segmentation. Default 10.
* **n_texton**: number of texton, or the number of clusters when doing rotation-invariant k-means over Gabor responses.


