

============ Obsolete ============

First we apply the classifiers to the images using `apply_classifiers_v3.py`. It is a multi-process code over sections. It generates scores on a sparse grid locations. Outputs are in `$SPARSE_SCORES_ROOTDIR`.

Step 2: Resample sparse scores.
Specify a resolution. Resample the score maps by the resolution.
Resampled score maps are used to generate score volumes and score map visualizations.

Step 3: Construct score volumes.
Specify a resolution. Load corresponding score maps. Stack the score maps up to form score volumes.
Script `construct_score_volume_v4.py` for one structure. Single-process program (?). Distribute structures over cluster.
Outputs are in `$VOLUME_ROOTDIR`. The volume as a 3D numpy array `volume` and the bounding box (xmin,xmax,ymin,ymax,zmin,zmax) `bbox`.
A lot of outputs involved, so it is better to use local /scratch.

Step 4: Visualize score maps (optional).
Specify a resolution. Load corresponding score maps. Generated visualizations are JPEG images at `$SCOREMAP_VIZ_ROOTDIR`. Heatmap is `plt.cm.hot`.
Script `visualize_scoremaps_v3.py`.

===================================