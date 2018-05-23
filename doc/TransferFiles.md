## Transfer files to/from birdstore

- Download folders: `scp -r <username>@birdstore.dk.ucsd.edu:<server_data_dir> <local_data_dir>`
- Download files: `scp <username>@birdstore.dk.ucsd.edu:<server_file_path> <local_data_dir>/`
- Upload folders: `scp -r <local_data_dir>/ <username>@birdstore.dk.ucsd.edu:<server_data_dir>`
- Upload files: `scp <local_file_dir>/ <username>@birdstore.dk.ucsd.edu:<server_data_dir>/`

## Transfer files to/from AWS S3

Method 1 (recommended): Use [CrossFTP](http://www.crossftp.com/).

Method 2: Using command-line tool
1. Install the aws command-line tool https://aws.amazon.com/cli/
2. Run "aws configure", enter the Access Key ID and Secret Access Key in the csv file (`datauser_credentials.csv`). Set region to "us-west-1".
3. To download images of a stack, run "aws s3 cp --recursive s3://mousebrainatlas-data/CSHL_data_processed/MD585/MD585_prep2_lossless_jpeg <local_folder>". 
4. To upload, run `aws s3 cp <local_filepath> s3://mousebrainatlas-data/<s3_filepath>`

Method 3: Using web console (This method cannot download the whole folder)
1. Go to https://mousebrainatlas.signin.aws.amazon.com/console
2. Login as 
username: datauser
password: <no passwords in github>
3. Choose "S3"
4. The data are in the bucket called "mousebrainatlas-data". 
5. Click on a file and click "Download".
