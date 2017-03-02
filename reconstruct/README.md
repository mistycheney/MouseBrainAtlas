The *reconstruct* folder contains code for reconstructing volumes from sections. The sections might be annotated or are score maps.

# Various volume definitions #

The annotation volume is formed by first aligning the image

Mesh vertex locations are 

## Construct Score Volume ##

Under folder `reconstruct`,

Run `construct_score_volumes_v2_distributed.ipynb`.
Script `construct_score_volume_v2.py`.

Score volumes are stored in `VOLUME_ROOTDIR`,
`<stack>/score_volumes/<stack>_down32_scoreVolume_<label>.bp`

This step takes 200 seconds.

## Compute Score Volume Gradient ##

Under folder `registration`,

Run `compute_gradient.ipynb`

Gradient files are stored in `VOLUME_ROOTDIR`,
`<stack>/score_volume_gradient/<stack>_down32_scoreVolume_<label>_[gx|gy|gz].bp`

This step is very fast.

## Construct Thumbnail Volume ##

Under folder `reconstruct`,
Run `construct_thumbnail_volume.ipynb`

Thumbnail volumes are stored in `VOLUME_ROOTDIR`,
`<stack>/<stack>_down32Volume.bp`
