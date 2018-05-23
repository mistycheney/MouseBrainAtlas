


# Volume

A volume is represented by:
- a 3-D array stored as `bp` file.
- a (3,) int array representing the origin of this array with respect to _wholebrain_ (see [Definition of frames]), stored as `txt` file.

## Volume type
Three volume types are defined, each with a different 3-d array data type:
- `annotationAsScore`: float, binary either 0 or 1
- `score`: float between 0 and 1
- `intensity`: uint8
