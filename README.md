### BrainSaliencyDetection

Project directory is `/oasis/projects/nsf/csd181/yuncong/Brain`

Executable scripts are at `/oasis/projects/nsf/csd181/yuncong/Brain/scripts`. It is recommended to run the code on gcn (instead of ion) because the numpy in yuncong's virtualenv is compiled with MKL (Intel Math Kernel Library) which is only installed on gcn.

Data are in `/oasis/projects/nsf/csd181/yuncong/ParthaData` and `/oasis/projects/nsf/csd181/yuncong/DavidData`

Output are in `/oasis/scratch/csd181/yuncong/output`

Example Workflow
-----

First log into any one of the 16 gcn computers. The hostnames of these computers are`gcn-20-x.sdsc.edu` where `x` is a number between 31 and 38 and between 41 and 48.

```shell
cd /oasis/projects/nsf/csd181/yuncong/Brain/scripts
```
Set up PATH for relavant executable and PYTHONPATH for cv2 package.
```shell
source /oasis/projects/nsf/csd181/yuncong/Brain/setup.sh
```

Run the feature extraction pipeline, for example,
```shell
# david data
python feature_extraction_pipeline_v2.py ../DavidData/RS155_x5/RS155_x5_0004.tif nissl324
# partha data
python feature_extraction_pipeline_v2.py ../ParthaData/PMD1305_region0_reduce2/PMD1305_region0_reduce2_0244.tif nissl324
```


*(needs update)* Then download results by running [`download_all2.sh`](https://gist.github.com/mistycheney/d92009bbb14b2951977d) on local machine.
```shell
# david data
./download_all2.sh /oasis/projects/nsf/csd181/yuncong/DavidData/x1.25/RS141_2_x1.25_z0.tif RS141_2_x1.25_z0_param10 output yuncong
# partha data
./download_all2.sh /oasis/projects/nsf/csd181/yuncong/ParthaData/PMD1305_region0_reduce2/PMD1305_region0_reduce2_0244.tif PMD1305_region0_reduce2_0244_param10 output yuncong
```
Just type `./download_all2.sh` to see the detailed help message.


**Note**: Currently, foreground mask detection does not work well for David's data. It is implemented in the function `foreground_mask` in `/oasis/projects/nsf/csd181/yuncong/Brain/scripts/utilities.py`.


Feature Extraction
-----

Feature extraction pipeline is implemented in the script `feature_extraction_pipeline_v2.py`.
Run `python feature_extraction_pipeline_v2.py -h` to see the detailed help message.


Data
----
**Data from David**: 

Data are stored at `/oasis/projects/nsf/csd181/yuncong/DavidData`.

Original data are 12 ndpi files. Each ndpi file contains 5 resolution levels. The script `split_all.sh` is used to split different levels of all images into seperate tif files. The tif files are stored in 5 directories corresponding to the 5 levels: `x0.078125`, `x0.3125`, `x1.25`, `x5`, `x20`. Here `x20` contains the images with the highest resolution.

Images are then manually segmented and stored in sub-directories such as `RS141_x5`. Images in each sub-directory have image index as suffix, e.g. `RS141_x5_0003.tif`.


**Data from Partha**:

Data are stored at `/oasis/projects/nsf/csd181/yuncong/ParthaData`.

The original data are 10 tar.gz files with names such as `PMD1305.tar.gz`. Each tarball contains a stack of images. Untarred jpeg2000 files from each tarball are stored in directories `PMD1305`, `PMD1328`, etc.

The other directories contains un-compressed subsets of the images at a particular resolution, in tif format. Each subset is referred to as a *dataset*. For example, `PMD1305_region0` is the dataset that includes images 240 through 250 in stack PMD1305. There is no enforced naming rules for a dataset. In this example, `region0` refers to a bounding box containing the brainstem. The dataset definition files are stored in `dataset_defs`.

To generate a new dataset, use the script `generate_dataset.py`. Just type `python generate_dataset.py -h` to see the detailed help message.

Output
-----

Outputs are stored in a sub-directory named `<image name>_param_<parameter id>`, under `/oasis/scratch/csd181/yuncong/output`.

`<suffix>` is one of:
* `features.npy`: Gabor filter responses. shape (`im_height`, `im_width`, `n_features`)
* `segmentation.npy`: same size as the image, each pixel is an integer indicating the index of the suerpixel it belongs to.
* `sp_dir_hist_normalized.npy`: energy distribution over different directions for each superpixel, shape (`n_superpixel`, `n_angle`)
* `sp_texton_hist_normalized.npy`: texton distribution for each superpixel, shape (`n_superpixel`, `n_texton`)
* `textonmap.npy`: same size as the image, each pixel is an integer from 0 to `n_texton-1` or -1 for background, shape (`im_height`, `im_width`)

* `segmentation.png`
* `textonmap.png`
* `texton_saliencymap.png`


Running the following command on your local machine returns a list of available results.
```shell
ssh <gordon username>@gcn-20-32.sdsc.edu 'ls -d /oasis/scratch/csd181/yuncong/output/*/'
``` 

One can issue the following command to download all generated images.
`scp <gordon username>@gcn-20-32.sdsc.edu:/oasis/scratch/csd181/yuncong/output/<image name>/*.png <image name>`.



<a name="param"></a> Parameters
-----

Parameter settings are stored as JSON files under the `params` sub-directory. Each JSON file specifies a particular set of parameters. They are named `param_<param id>.json`. `param id` can be any string, for example `nissl324`. The default parameter file is named `param_default.json`.

Parameter fields are allowed to be NaN, in which case the values will be replaced by the corresponding values in the default setting.

#### Gabor filter bank parameters ##
* **param_id**: an integer id for this set of parameters, default parameter is id 0 (change to string)
* **min_wavelen**: minimum wavelength of Gabor filter, in number of pixels, default 5
* **max_wavelen**: maximum wavelength of Gabor filter, in number of pixels, default 40
* **freq_step**: multiply factor to the next frequency, default 2 (same as factor for wavelength)
* **theta_interval**: interval of Gabor filter orientations, in degrees, default 15.
* **bandwidth**: wave-length/std-of-gaussian, doubling the value corresponds to halving the size of the Gaussian kernel in each dimension (kernel is always spherical). Default 1.0 = 3/2 wavelengths within the window.

#### Segmentation parameters ##
* **n_superpixels**: desired number of superpixels in the over-segmentation, default 100. Large number leads to smaller superpixels.
* **slic_compactness**: larger value leads to super-pixels that are more square. Default 5.
* **slic_sigma**: width of Gaussian kernel used in pre-smoothing before segmentation. Default 10.

#### Texton K-Means parameters ##
* **n_texton**: number of texton, or the number of clusters when doing rotation-invariant k-means over Gabor responses.
* **n_sample**: number of samples to use at each iteration of Kmeans. default 10,000 
* **n_iter**: number of iterations of Kmeans. default 10

#### Detector parameters ##
* **n_models**: number of models to detect. default 10
* **beta**: a number that controls how close the significance under new weight is to zero. defaut 1.0
* **frontier_contrast_diff_thresh**: relative entropy region growing will stop incrementing threshold as long as the difference between the current and the previous frontier contrasts exceeds this value. default 0.2
* **lr_grow_thresh**: include a neighbor superpixel into current cluster if the likelihood ratio P(superpixel|model)/P(superpixel|null) of this superpixel exceeds this value. default 0.1
* **lr_decision_thresh**: when applying learned models, if the likelihood ratio of a superpixel P(superpixel|model)/P(superpixel|null) is smaller than this value, this suerpixel is classified as NULL. default 0.3
 

