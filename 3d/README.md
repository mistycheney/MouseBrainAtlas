`visualize_annotated_brains`: reconstruct 3-D structures as binary volumes from annotation contours; convert structures to meshes.

`render_shell_from_tissue_masks`: reconstruct the tissue outline as a 3-D shell, based on tissue masks.




Domains:

- atlas_space_structure_meshes: wrt "canonical centroid"
"canonical centroid" is defined wrt to the aligned uncropped volume of MD589
- annotation volume meshes: wrt aligned cropped volume
- annotation surround meshes: wrt atlas volume

For structure meshes, position is defined with respect to the cropped thumbnail volume (this for the x and y dimensions are the same as the score volume, though the score volume has smaller z dimension).

For shell meshes, position is defined with respect to the aligned uncropped volume.
