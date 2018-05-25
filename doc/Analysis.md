# Population variability

- deviation vector from mean position
- RMS variation distance in 3D, as well as in rostral-caudal, dorsal-ventral and lateral medial axes.
- non-isotropy of variation

`registration/update_atlas.ipynb`

# Registration Error

### Registration error compared to ChAT labels

- centroid position error
- Jaccard index between aligned atlas and ChAT

Computed for after local registration.

`registration/evaluate_registration_metrics_v2_compute_deviation_vs_ChAT.ipynb`

### Registration error compared to expert annotations

- centroid position error
- Jaccard index between aligned atlas and expert annotation.
- voxel value difference

Computed for both after global registration, and after local registration.

`registration/evaluate_registration_metrics_v2_compute_deviation_vs_expert.ipynb`

# Human Correction

### Quantify human correction
`registration/analyze_human_correction.ipynb`
  
### Global to local difference
`registration/update_atlas.ipynb`

### local to human difference
`registration/update_atlas.ipynb`

# Registration Confidence

- z-score for global and local registration
- peak width in most uncertain direction:
- peak width in most certain direction:

`registration/measure_confidence_v4.ipynb`

`/home/yuncong/Dropbox/BrainProjectFiguresByTopic/Registration/confidence/measurements/peakradius_max_normalized_allstacks_allstructures_allsteps_allpools.json`


