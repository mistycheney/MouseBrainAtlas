### Convert to JPEG
Convert to JPEG: 5.48 seconds. (without uploading to s3): 6s * 300 sections = 30 mins

### Convert from RGB to gray
Load: 8.50 seconds.
Convert RGB to gray: 14.33 seconds. # 0 if only taking the blue channel
Save: 5.19 seconds. # 2.5 seconds if not uploading to s3

### For nissl data
* raw -> thumbnail
* thumbnail -> prep1_thumbnail
* raw -> prep1_raw
* prep1_raw -> prep5_raw (optional):
* prep1_thumbnail -> prep5_thumbnail:
* prep5_raw -> prep2_raw
* prep2_raw -> prep2_raw_gray: 30s * ~300 sections = 150 mins
* prep2_raw_gray -> prep2_raw_grayJpeg: 20s * ~300 sections = 100 mins
* prep2_raw -> prep2_raw_jpeg: 20s * ~300 sections = 100 mins

### For neurotrace data
* raw_Ntb -> thumbnail_Ntb
* thumbnail_Ntb -> thumbnail_NtbNormalized
* thumbnail_NtbNormalized -> prep1_thumbnail_NtbNormalized
* raw_Ntb -> raw_NtbNormalizedAdaptive
* raw_NtbNormalizedAdaptive -> prep1_raw_NtbNormalizedAdaptive
* prep1_raw_NtbNormalizedAdaptive -> prep5_raw_NtbNormalizedAdaptive:
* prep1_thumbnail_NtbNormalized -> prep5_thumbnail_NtbNormalized:
* prep5_raw_NtbNormalizedAdaptive -> prep2_raw_NtbNormalizedAdaptiveInvertedGamma
* prep2_raw_NtbNormalizedAdaptiveInvertedGamma -> prep2_raw_NtbNormalizedAdaptiveInvertedGammaJpeg

### Images to score volume

* locate patches: 0.02 seconds

* Load pre-computed features: 0.07 seconds
* No pre-computed features found... computing from scratch.
  * Load image: 20.64 seconds.
  * Crop patches: 0.09 seconds.
  * Extract patches: 20.74 seconds
  * Compute features: 4.67 seconds
  * Save features: 2.34 seconds

* Load background image: 19.82 seconds (Load background image: 1.48 seconds if already exist)
* Rescale background image to output resolution: 0.17 seconds
* Predict scores 7N: 0.01 seconds
* Resample scoremap 7N: 0.08 seconds
* Load and rescale background image: 0.00 seconds
* Generate scoremap overlay: 0.04 seconds.
* Scoremap size does not match background image size. Need to resize: 0.06 seconds.
* Generate scoremap overlay image 7N: 0.12 seconds

### Compute features for entire images
* locate patches: 0.73 seconds
* No pre-computed features found... computing from scratch.
* Load image: 3.15 seconds.
* Crop patches: 0.69 seconds.
* Extract patches: 3.85 seconds (23960, 1, 224, 224)
* Compute features: 77.74 seconds (15.06 seconds if using 8 GPUs)
* (total) Compute features at one section, multiple locations: 81.60 seconds (16.85 seconds if using 8 GPUs)
* Save features: 2.64 seconds
-----------------
One stack: 17s * ~300 sections = 85 mins


