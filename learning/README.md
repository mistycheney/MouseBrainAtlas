This folder contains code related to learning texture detectors. This include extracting patches, transforming to feature vectors, and training SVM classifiers.

## Feature extraction ##

`extract_test_features_dnn.ipynb` runs on the Workstation equipped with GPU. It calls MXNet to compute features for patches through forward pass over a deep neural network.

The generated features are place in `PATCH_FEATURES_ROOTDIR`. By default this is
`/media/yuncong/BstemAtlasData/CSHL_patch_features_Sat16ClassFinetuned_v2`.
The
`<stack>/<fn>_lossless_alignedTo_<anchor_fn>_cropped`. The features as n x 1024 array are stored in `<fn>_lossless_alignedTo_<anchor_fn>_cropped_features.hdf`. The locations of corresponding n patches are stored in `<fn>_lossless_alignedTo_<anchor_fn>_cropped_patch_locations.txt`.

This step takes 80 seconds per section (~25k patches).

## Train ##

Code for training classifiers is in `train_classifier_v2.ipynb`.
It works for both regular nissl and Neurotrace blue.

Store in `SVM_DIR`. By default this is `CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers`. The classifiers are `<label>_svm.pkl`.

`Cluster executable`
`svm_v2.py`

Performance of trained classifiers can be visually analyzed in `test_classifier_performance.ipynb`.

## Pipeline ##

The pipeline takes as input a stack whose DNN features are available.
It goes through the following stages:
- apply classifiers, generate score volumes
- global alignment with atlas
- local alignment with atlas

The pipeline can be controlled by `pipeline_features_to_scoremaps.ipynb`. It launches the proper
executable for each task on the computing cluster.

## Predict ##

Run `svm_v2.ipynb`

Load pre-trained classifiers. Apply every classifier to every feature file.

Predicted sparse scores are stored in `PREDICTIONS_ROOTDIR`.
By default this is `CSHL_patch_Sat16ClassFinetuned_v2_predictions`.

Under `<stack>/<fn>_lossless_alignedTo_<anchor_fn>_cropped`,
each sparse score file, as n x 1 array, is `<fn>_lossless_alignedTo_<anchor_fn>_cropped_<label>_sparseScores.hdf`

This step takes 900 seconds per stack.

## Interpolate ##

Run `interpolated_scoremaps_v2_distributed.ipynb`.

Script is `interpolated_scoremaps_v2.py`.
`interpolated_scoremaps_v2.py stack first_sec last_sec`.

Dense scoremaps are stored in `SCOREMAPS_ROOTDIR`. By default `CSHL_lossless_scoremaps_Sat16ClassFinetuned_v2`.
Under `<stack>/<fn>_lossless_alignedTo_<anchor_fn>_cropped`,
each dense score file is `<fn>_lossless_alignedTo_<anchor_fn>_cropped_<label>_denseScoreMap.hdf`,
and associated bounding box file `<fn>_lossless_alignedTo_<anchor_fn>_cropped_<label>_denseScoreMap_interpBox.txt`.

This step takes ? seconds per stack.

## Visualize ##

Run `visualize_scoremaps_v2_distributed.ipynb`.

Script is `visualize_scoremaps_v2.py`
`visualize_scoremaps_v2.py stack -b first_sec -e last_sec -a`

Score map visualizations are stored in `SCOREMAPVIZ_ROOTDIR`.
By default is `CSHL_scoremap_viz_Sat16ClassFinetuned_v2`.

Under `<label>/<stack>/<fn>_alignedTo_<anchor_fn>_scoremapViz_<label>.jpg`.

This step takes 500 seconds per stack.


## Evaluation / Analysis ##

The algorithm's goal is texture classification. The tasks are to separate textures inside a structure from textures outside the structure. Because we already have a strong location prior, we don't need strong texture score signal at places far from the structure's true location, so we take as negative examples only textures at the surrounding of the structure.

If annotations are available, we can compute the true positive, true negative, false positive and false negative rates of classifying each structure. We plot ACC as a function of the margin of the surrounding where negative samples are collected.
