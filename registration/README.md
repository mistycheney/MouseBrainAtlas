The *registration* folder contains code for volume-to-volume registration. This includes global alignment and individual structure alignment.

Two attempts for more fine-grained contouring of structures:
- `fit_contour_to_scoremap_graphcut.ipynb`: Apply graph-cut to an ROI of scoremap. Use structure-wise aligned contours to identify the right connected component in case multiple CCs are generated.
- `fit_contour_to_scoreVolume_inSection.ipynb`: Find the best 2D affine transform of structure-wise aligned contours on every section.

# Compute Gradients
- `compute_gradient.py`: compute gradients, save as `[stack]_scoreVolume_[label]_[gx|gy|gz].bp` files.

# Registration

## Global Alignment

- `run_global_registration_distributed.sh`: wrapper script for distributing global alignment of multiple brains over multiple machines. It calls `global_affine_alignment.py`.

- `global_affine_alignment.py`: global alignment. Experimental notebook is `align_3d_v2_affine_atlas.ipynb`.

Final version of the notebook is `align_atlas_global.ipynb`


### Timing

*Compute transforms*
- load gradient: 1s x 28 structures = 28s
- grid search: 60s
- gradient descent: 10s x 80 iteration = 800s


## Align two annotated Volumes
- `align_3d_v2_affine_annotations.ipynb`

## Individual Structure Alignment

- `run_align_individual_landmarks_distributed.sh`: wrapper script for running individual structure alignment for multiple brains. It calls `align_individual_landmarks.py`.

- `align_individual_landmarks.py`: individual structure alignment. Experimental notebook is `align_3d_v2_affine_atlas_individual_lie_twoSides.ipynb`.

Final version of the notebook is `align_atlas_individual.ipynb`

`fit_altlas_structure_to_subject_v2.ipynb`

### Timing

*Compute transforms*
- load gradient: 2s
- grid search: 0.3s x 30 iterations = 9s
- gradient descent: 0.04s x 200 iterations = 8s

overall x 50 structures (sided) =

*Transform volumes according to computed parameters*
- x 50 structures (sided) =

*Generate overlay images*
- x 250 sections =

## Compute Hessians
1st version

2nd version
- `compute_global_alignment_hessian.ipynb`
- `compute_individual_alignment_hessian.ipynb`
3rd version
- `hessian_v2`
