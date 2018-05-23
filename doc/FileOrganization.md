Image data
==========

Processed images (aligned, cropped) are stored in AWS S3. Note that the filenames do not reflect the correct order, you will need the sorted file list (`/mousebrainatlas-data/CSHL_data_processed/MD585/MD585_sorted_filenames.txt`).

Here are the instructions for downloading one stack MD585 (~300 sections) from AWS. 

Method 1: Using command-line tool (recommended)
1. Install the aws command-line tool https://aws.amazon.com/cli/
2. Run "aws configure", enter the Access Key ID and Secret Access Key in the csv file (`datauser_credentials.csv`). Set region to "us-west-1".
3. To download images of a stack, run "aws s3 cp --recursive s3://mousebrainatlas-data/CSHL_data_processed/MD585/MD585_prep2_lossless_jpeg <local_folder>". 

Method 2: Using web console (This method cannot download the whole folder)
1. Go to https://mousebrainatlas.signin.aws.amazon.com/console
2. Login as 
username: datauser
password: <no passwords in github>
3. Choose "S3"
4. The data are in the bucket called "mousebrainatlas-data". 
5. Click on a file and click "Download".

**Yoav:** This data needs to be reorganized in an image storage to allow fast retrieval of individual sections and parts of sections.

## reconstructed volumes or virtual sections
Collection of images representing virtual sections in all three directions (sagittal, coronal and horizontal).

Meta data
===

Annotations
-----------

There are three types of annotations (points, 2D polygons, 3D volumes). Each stack usually has one of each type.
They are stored as HDF tables. Each row represents one point/polygon/volume.

For point and 2D contour annotation files, the column names are:
```
Index([u'class', u'creator', u'downsample', u'edits', u'id', u'label_position',
       u'name', u'orientation', u'parent_structure', u'section', u'side',
       u'side_manually_assigned', u'time_created', u'type', u'vertices',
       u'filename'],
```

- `class`: "contour" or "neuron"
- `creator`: username of the creator
- `downsample`: the downsample factor the vertices are defined on
- `edits`: the list of edits made on this contour
- `id`: a random uuid for this contour
- `label_position`: the position of the structure name text item relative to the whole image
- `name`: unsided name of this structure
- `orientation`: sagittal, coronal or horizontal
- `parent_structure`: currently not used
- `section`: the section number
- `side`: L or R
- `side_manually_assigned`: True if the side is confirmed by human; False if the side is automatically inferred.
- `time_created`: the time that the contour is created
- `type`: "intersected" if this contour is the result of interpolation or "confirmed" if confirmed by human
- `vertices`: vertices of a polygon. (n,2)-ndarray.
- `filename`: the file name of the section.


