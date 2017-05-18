Let's upload the files needed for aligning one subject MD593 with the atlas.

- Folder for score volumes
`VOLUME_ROOTDIR = $CSD395/CSHL_volumes2`.
Under this, we are intereted in:
  - Atlas
`atlas_on_MD589/score_volumes`
  - Subject
`MD593/score_volumes`
  - Subject gradients
`MD593/score_volume_gradients`

- Folder for registration parameters
`atlasAlignParams_rootdir`.
This is where the program will write the output parameter files.
