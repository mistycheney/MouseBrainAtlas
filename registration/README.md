The *registration* folder contains code for volume-to-volume registration. This includes global alignment and individual structure alignment.

## Compute Hessians
- `compute_global_alignment_hessian.ipynb`
- `compute_individual_alignment_hessian.ipynb`

## Batch Processing Scripts

### Global Alignment

- `run_global_registration_distributed.sh`: wrapper script for distributing global alignment of multiple brains over multiple machines. It calls `global_affine_alignment.py`.
- `global_affine_alignment.py`: global alignment. Experimental notebook is `align_3d_v2_affine_individual_lie_twoSides_hessian.ipynb`.

### Individual Structure Alignment

- `run_align_individual_landmarks_distributed.sh`: wrapper script for running individual structure alignment for multiple brains. It calls `align_individual_landmarks.py`
