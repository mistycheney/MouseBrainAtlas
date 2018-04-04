# Preprocessing

map from prep a to prep b


# Register Annotated Brains

`align_annotated_brains_v6.ipynb`

This uses the class `Aligner`.


# Build/Update Atlas from globally aligned brains.

- Compute mean centroid of each structure in co-aligned space.
- Rectify co-aligned space to define the canonical atlas space.
- Compute mean shape of each structure (only do this at initialization, not in later updates)

