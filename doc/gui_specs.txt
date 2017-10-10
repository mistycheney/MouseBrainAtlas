# Requirements specific to Mouse Brain image annotation

- [Image Manipulation](#image-manipulation)
- [Annotation](#annotation)
- [3D Structure Manipulation](#3d-structure-manipulation)
- [Database Association: Brain structure names](#database-association-brain-structure-names)

### Image Manipulation
The user should be able to load and manipulate a large number of section images (> 300 sections for one stack).

#### User actions to support
- Show a gallery of section thumbnails for quick browsing.
- Once the user selects a section, show the full-resolution image (take as much as screen space as possible).
  - Pan/zoom
  - Adjust contrast (on-the-fly perferrably)
  - Allow switching on/off each of R/G/B channels (useful for fluorescent images)
- Switch to previous/next image. While doing so, keep the same viewport (i.e. same position, same zoom level).

### Annotation
Annotations are in the form of graphical elements overlaid on the image.
We need three types of overlay elements:
- **polygon**: A collection of connected line segments. Represent structure boundaries. Each Polygon has several attributes (name, creator, creation time, section number, list of vertex coordinates etc.)
- **point**: A filled circle. Represent vertices of a polygon. Each point is associated with a polygon. 
- **text**: A text item. Represent brain structure names. Each text item is associated with a polygon.

#### User actions to support
- Move/add/delete polygons
- Move/add/delete points 
- Move/add/delete texts
- Turn on/off polygon/vertices/texts
- Change appearance of elements (e.g. increase thickness of polygon lines, change color of vertices)

### 3D Structure Manipulation
- 3-view sync
- User updates 3D reconstruction of a structure, front-end updates virtually sectioned contours in 3-view

### Database Association: Brain structure names
A brain structure has a full name and an abbreviation.

#### User actions to support
- A user should be able to create new names
- A user can also choose name from a given list ([MouseBrainAtlas/gui/structure_names.txt](https://github.com/mistycheney/MouseBrainAtlas/blob/master/gui/structure_names.txt)) Ideally when the user types a word, he will be given a shortlist of names that have this word in it and he can choose from this shortlist. Even better if the interface can reflect the hierarchy of structure names (e.g. "brainstem" includes "facial motor nucleus" and "trigeminal motor nucleus").
DOUBLE DEALS
PROFILE
