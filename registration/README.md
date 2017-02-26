The *registration* folder contains code for volume-to-volume registration. This includes global alignment and individual structure alignment.

Two attempts for more fine-grained contouring of structures:
- `fit_contour_to_scoremap_graphcut.ipynb`: Apply graph-cut to an ROI of scoremap. Use structure-wise aligned contours to identify the right connected component in case multiple CCs are generated.
- `fit_contour_to_scoreVolume_inSection.ipynb`: Find the best 2D affine transform of structure-wise aligned contours on every section.

# Registration

## Registration Setting ##

1:
Preceeding warping: None
Score volume: classifier setting 2
Transform: global affine (One parameter set)

2:
Preceeding warping: warping setting 1
Score volume: classifier setting 2
Transform: structure-wise rigid (One parameter set for each structure)
Regularization = 0

3:
Upstream warping: warp setting 2
Score volume: classifier setting 2
Transform: global affine (One parameter set) = the inverse of parameters found by warp setting 1

4:
Upstream warping: warp setting 1
Score volume: classifier setting 2
Transform: structure-wise rigid (One parameter set for each structure)
Regularization = np.array([1e-6, 1e-6, 1e-6])



## Global Registration

`global_registration_v3.ipynb`
`global_registration_v3.py`

## Align two annotated Volumes
- `align_3d_v2_affine_annotations.ipynb`

## Structure-wise Registration

`local_registration_v3.ipynb`
`local_registration_v3.py`

### Timing

Global
- load gradient: 1s x 28 structures = 28s
- grid search: 60s
- gradient descent: 10s x 80 iteration = 800s

Local
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
