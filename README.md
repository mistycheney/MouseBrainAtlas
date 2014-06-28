BrainSaliencyDetection
======================

Project directory is `/oasis/projects/nsf/csd181/yuncong/Brain`

Executable scripts are at `/oasis/projects/nsf/csd181/yuncong/Brain/scripts`.
This currently includes two scripts: `feature_extraction_pipeline.py` and `generate_dataset.py`
It is recommended to run the code on gcn (instead of ion).


Data are in `/oasis/projects/nsf/csd181/yuncong/ParthaData` and `/oasis/projects/nsf/csd181/yuncong/DavidData`

Output are in `/oasis/scratch/csd181/yuncong/output`


Example Workflow
-----

Below are the steps to process an image from DavidData.

First log into Gordon
<pre>
cd Brain/scripts
python feature_extraction_pipeline.py ../../DavidData/x1.25/RS141_2_x1.25_z0.tif 10
</pre>

Then on local machine
<pre>
./download_all2.sh /oasis/projects/nsf/csd181/yuncong/DavidData/x1.25/RS141_2_x1.25_z0.tif RS141_2_x1.25_z0_param10 output yuncong
</pre>


**Note**: Currently, background removal does not work well for David's data. Implementation of background removal is in `/oasis/projects/nsf/csd181/yuncong/Brain/scripts/utilities.py`.

Feature Extraction
-----

Feature extraction pipeline is implemented in the script `feature_extraction_pipeline.py`.
Run `python feature_extraction_pipeline.py -h` to see the detailed help message.

Output will be generated in the sub-directory `<dataset name>_<image index>_param<parameter id>` under the output path. One can download the output by running [`download_all2.sh`](https://gist.github.com/mistycheney/d92009bbb14b2951977d) on the local machine.
Just type `./download_all2.sh` to see the usage.

Data
----
**Data from Partha**:

Data are stored at `/oasis/projects/nsf/csd181/yuncong/ParthaData`.

The original data are 10 tar.gz files with names such as `PMD1305.tar.gz`. Each tarball contains a stack of images. Untarred jpeg2000 files from each tarball are stored in directories `PMD1305`, `PMD1328`, etc.

The other directories contains un-compressed subsets of the images at a particular resolution, in tif format. Each subset is referred to as a *dataset*. For example, `PMD1305_region0` is the dataset that includes images 240 through 250 in stack PMD1305. There is no enforced naming rules for a dataset. In this example, `region0` refers to a bounding box containing the brainstem. The dataset definition files are stored in `dataset_defs`.

To generate a new dataset, use the script `generate_dataset.py`. Just type `python generate_dataset.py -h` to see the detailed help message.

**Data from David**: 

Data are stored at `/oasis/projects/nsf/csd181/yuncong/DavidData`.

Original data are 12 ndpi files. Each ndpi file contains 5 resolution levels. The script `split_all.sh` is used to split different levels of all images into seperate tif files. The tif files are stored in 5 directories corresponding to the 5 levels: `x0.078125`, `x0.3125`, `x1.25`, `x5`, `x20`. Here `x20` contains the images with the highest resolution.

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


Running `ssh <gordon username>@ion-21-14.sdsc.edu 'ls -d Brain/output/*/'` from local machine returns a list of available results.


<a name="param"></a> Parameters
-----

Parameter settings are specified in `params.csv` under the project directory. 

To make editing the parameters easier, you can also modify [this google spreadsheet](https://docs.google.com/spreadsheets/d/1S189da_CxzC3GKISG3hZDG0n7mMycC0v4zTiRJraEUE/edit), and then use [this script](https://gist.github.com/mistycheney/be1f758bfcd5f852c9b5#file-sync_params_google_spreadsheet-py) to download the corresponding csv file to overwrite `params.csv`.

#### Gabor filter bank parameters ##
* **param_id**: an integer id for this set of parameters, default parameter is id 0
* **min_wavelen**: minimum wavelength of Gabor filter, in number of pixels, default 5
* **max_wavelen**: maximum wavelength of Gabor filter, in number of pixels, default 40
* **freq_step**: multiply factor to the next frequency, default 2 (same as factor for wavelength)
* **theta_interval**: interval of Gabor filter orientations, in degrees, default 15.
* **bandwidth**: larger value means narrower Gaussian in spatial domain, thus smaller kernel size. default 1.0

#### Segmentation parameters ##
* **n_superpixels**: desired number of superpixels in the over-segmentation, default 100. Large number leads to smaller superpixels. Default 100.
* **slic_compactness**: larger value leads to more square segmentation. Default 5.
* **slic_sigma**: width of Gaussian kernel used in pre-smoothing before segmentation. Default 10.

#### Texton K-Means parameters ##
* **n_texton**: number of texton, or the number of clusters when doing rotation-invariant k-means over Gabor responses.
* **n_sample**: number of samples to use at each iteration of Kmeans. default 10,000 
* **n_iter**: number of iterations of Kmeans. default 10


