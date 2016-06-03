The *registration* folder contains code for volume-to-volume registration. This includes global alignment and individual structure alignment.


## Construct Volumes
- `compute_score_volume.ipynb`: generates `[stack]_scoreVolume_[label].bp` and `[stack]_scoreVolume_limits.txt`.
- `compute_thumbnail_volume.ipynb`:
- `compute_score_volume.ipynb`:

## Compute Gradients
- `compute_gradient.py`: compute gradients, save as `[stack]_scoreVolume_[label]_[gx|gy|gz].bp` files.

## Registration

### Global Alignment

- `run_global_registration_distributed.sh`: wrapper script for distributing global alignment of multiple brains over multiple machines. It calls `global_affine_alignment.py`.

- `global_affine_alignment.py`: global alignment. Experimental notebook is `align_3d_v2_affine_atlas.ipynb`.

Final version of the notebook is `align_atlas_global.ipynb`

### Align two annotated Volumes
- `align_3d_v2_affine_annotations.ipynb`

### Individual Structure Alignment

- `run_align_individual_landmarks_distributed.sh`: wrapper script for running individual structure alignment for multiple brains. It calls `align_individual_landmarks.py`.

- `align_individual_landmarks.py`: individual structure alignment. Experimental notebook is `align_3d_v2_affine_atlas_individual_lie_twoSides.ipynb`.

Final version of the notebook is `align_atlas_individual.ipynb`

## Compute Hessians
- `compute_global_alignment_hessian.ipynb`
- `compute_individual_alignment_hessian.ipynb`
