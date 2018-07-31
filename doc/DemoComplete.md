## Preprocess 
- Run `download_demo_data_preprocessing.py --step 1` to download an example JPEG2000 image.
- Run `jp2_to_tiff.py <brain> <input_spec>`. This will create `DEMO999_raw/` which contains `DEMO999-F148-2017.06.08-02.58.49_MD662_1_0442_raw.tif`.
- Run `download_demo_data_preprocessing.py --step 2`
- 
- `$ ./align.py DEMO999 `
- 

## Compute patch features
- Run `download_demo_data_compute_features.py`
- `$ ./compute_features_demo.py DEMO999 --section 151 --version NtbNormalizedAdaptiveInvertedGamma`

## Generate probability volumes
- Run `download_demo_data_scoring.py`
- `$ ./from_images_to_score_volumes_demo.py DEMO999 799 --structure_list [\"3N, 4N, 12N\"]`
