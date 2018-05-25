# Preprocessing

## Convert raw data to TIFF
jp2 -> raw: 
raw -> raw_Ntb:

### For thionin data
* raw -> thumbnail
* **Compute tranforms using thumbnail**
* thumbnail -> prep1_thumbnail
* **Supply prep1_thumbnail_mask**
* prep1_thumbnail_mask -> thumbnail_mask
* raw -> prep1_raw
* **Compute prep5 (alignedWithMargin) cropping box based on prep1_thumbnail_mask**
* prep1_raw -> prep5_raw:
* prep1_thumbnail -> prep5_thumbnail:
* prep1_thumbnail_mask -> prep5_thumbnail_mask:
* **Specify prep2 (alignedBrainstemCrop) cropping box**
* prep5_raw -> prep2_raw
* prep5_thumbnail -> prep2_thumbnail
* prep5_thumbnail_mask -> prep2_thumbnail_mask
* prep2_raw -> prep2_raw_gray: 30s * ~300 sections = 150 mins
* prep2_raw_gray -> prep2_raw_grayJpeg: 20s * ~300 sections = 100 mins
* prep2_raw -> prep2_raw_jpeg: 20s * ~300 sections = 100 mins

_prep2_raw_gray_ are used for structure detection.
_prep5_raw_ will be published online.

### For Neurotrace data
* raw_Ntb -> thumbnail_Ntb: 11s/section
* thumbnail_Ntb -> thumbnail_NtbNormalized: 0.1s/section
* **Compute transforms using thumbnail_NtbNormalized**
* **Supply prep1_thumbnail_mask**
* prep1_thumbnail_mask -> thumbnail_mask
* raw_Ntb -> raw_NtbNormalizedAdaptiveInvertedGamma (**brightness correction**)
* **Compute prep5 (alignedWithMargin) cropping box based on prep1_thumbnail_mask**
* raw_NtbNormalizedAdaptiveInvertedGamma -> prep5_raw_NtbNormalizedAdaptiveInvertedGamma: ~1.5min/section * 300 sections = 7.5 hrs
* thumbnail_NtbNormalized -> prep5_thumbnail_NtbNormalized: 70s/stack (8 threads)
* prep5_raw_NtbNormalizedAdaptiveInvertedGamma -> prep5_thumbnail_NtbNormalizedAdaptiveInvertedGamma: 5s/section
* **Specify prep2 (alignedBrainstemCrop) cropping box**
* prep5_raw_NtbNormalizedAdaptiveInvertedGamma -> prep2_raw_NtbNormalizedAdaptiveInvertedGamma: 1500s/stack (4 threads)
* prep2_raw_NtbNormalizedAdaptiveInvertedGamma -> prep2_raw_NtbNormalizedAdaptiveInvertedGammaJpeg: 14s/section

## Convert to JPEG
- Convert to JPEG: 5.48 seconds. (without uploading to s3): 6s * 300 sections = 30 mins

## Convert from RGB to gray
- Load: 8.50 seconds.
- Convert RGB to gray: 14.33 seconds. # 0 if only taking the blue channel
- Save: 5.19 seconds. # 2.5 seconds if not uploading to s3

## Brightness correction
- Load image: 23.39 seconds.
- Rescale mask: 24.36 seconds.
- Compute mean/std for sample regions: 8.19 seconds.
- Interpolate mean map: 6.42 seconds.
- Scale up mean map: 14.60 seconds.
- Interpolate std map: 6.18 seconds.
- Scale up std map: 17.01 seconds.
- Normalize: 9.15 seconds.
- Save float version: 6.89 seconds.
- Rescale to uint8: 14.76 seconds.

# Detection

## Compute features for entire images
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

## Images to score volume

For each section:
* locate patches: 0.02 seconds
* Load pre-computed features: 1 seconds
* No pre-computed features found... computing from scratch.
 * Load image: 20.64 seconds.
 * Crop patches: 0.09 seconds.
 * Extract patches: 20.74 seconds
 * Compute features: 4.67 seconds
 * Save features: 2.34 seconds
* Load background image: 19.82 seconds (Load background image: 2.5 seconds if already exist)
* Rescale background image to output resolution: 3.08 seconds
* Predict scores 7N: 0.01 seconds
* Resample scoremap 7N: 0.08 seconds
* Load and rescale background image: 0.00 seconds
* Generate scoremap overlay: 0.04 seconds.
* Scoremap size does not match background image size. Need to resize: 0.06 seconds.
* Generate scoremap overlay image 7N: 0.12 seconds
* Save scoremap: 0.01 seconds
* Save scoremap viz: 0.04 seconds
----------
* Images to volume:
* Save score volume:
* Compute gradient:
* Save gradient:

Overall, if features are pre-computed and background images already exist, 8 s/section * 300 sections = 40 mins.

# Registration

## Simple global registration

negligible

## Local registration



