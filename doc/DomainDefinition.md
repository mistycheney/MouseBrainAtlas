# 2-D image frames
- id=1 (alignedPadded): section-to-section aligned, with large paddings.
- id=5 (alignedWithMargin): tightly crop over full tissue area with fixed small margin on all four sides.
- id=2 (alignedBrainstemCrop): crop only the brainstem area (from caudal end of thalamus to caudal end of trigeminal caudalis, from top of superior colliculus to bottom of brain)

# 3-D volume frames
- **wholebrain**: 
- **brainstemXYfull**:
- **wholebrainXYcropped**: stack all prep2 images starting from section 1 (including ones that are not available but occupy an index slot). Origin is before potential margin.
- **brainstem**
- **brainstemXYFullNoMargin**
- **wholebrainWithMargin**: stack all available prep5 images. When forming the volume, image margins are subsumed into the origin's z-coordinate.
- **brainstemWithMargin**

Also see [Domain definition](https://docs.google.com/presentation/d/1o5aQbXY5wYC0BNNiEZm7qmjvngbD_dVoMyCw_tAQrkQ/edit#slide=id.g2d31ede24d_0_0).
