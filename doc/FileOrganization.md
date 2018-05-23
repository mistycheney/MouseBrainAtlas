This page describes the location of the raw input and intermediate data.

Data Path
============

- `REPO_DIR`: default to `/home/yuncong/Brain/`
- `DATA_DIR`: default to `CSHL_data_processed`
- `VOLUME_ROOTDIR`: 
- `MESH_ROOTDIR`:
- `REGISTRATION_PARAMETERS_ROOTDIR`:
- `PATCH_FEATURES_ROOTDIR`:
- `ANNOTATION_ROOTDIR`:
- `CLF_ROOTDIR`:
- `SCOREMAP_ROOTDIR`:
- `SCOREMAP_VIZ_ROOTDIR`:
- `REGISTRATION_VIZ_ROOTDIR`
- `MXNET_MODEL_ROOTDIR`


Data are stored in two places:
- Lab server `birdstore.dk.ucsd.edu` under `/brainbucket/data/Active_Atlas_Data/`. 
  - Raw images are stored in folders that correspond to the scanner used, e.g. `UCSD_AxioScanner/CSHL2_2018-04-04`.
  - Processed images and other data are stored in `CSHL_data_processed` and other folders.
- AWS S3 bucket `mousebrainatlas-rawdata` and `mousebrainatlas-data`.
  - Raw images are stored in bucket `mousebrainatlas-rawdata`.
  - Processed images and other data are stored in bucket `mousebrainatlas-data`.


(## Reconstructed volumes or virtual sections
Collection of images representing virtual sections in all three directions (sagittal, coronal and horizontal).)
