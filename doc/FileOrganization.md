This page describes the location of the raw input and intermediate data.

The storage locations of different types of data are specified by the following global variables in `global_setting.py`:

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

All data are stored relative to these root directories. The exact path for specific data is provided in the description of the system process that generates the data.

Data are stored long-term in two places:
- Lab server `birdstore.dk.ucsd.edu` under `/brainbucket/data/Active_Atlas_Data/`. 
  - Raw images are stored in folders that correspond to the scanner used, e.g. `UCSD_AxioScanner/CSHL2_2018-04-04`.
  - Processed images and other data are stored in `CSHL_data_processed` and other folders.
- AWS S3 bucket `mousebrainatlas-rawdata` and `mousebrainatlas-data`.
  - Raw images are stored in bucket `mousebrainatlas-rawdata`.
  - Processed images and other data are stored in bucket `mousebrainatlas-data`.

Refer to [this page](TransferFiles.md) on how to transfer data between lab server or S3 and the local workstation.


(## Reconstructed volumes or virtual sections
Collection of images representing virtual sections in all three directions (sagittal, coronal and horizontal).)
