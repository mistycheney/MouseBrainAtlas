# Tips #
- The nodewatcher running on the compute nodes is too aggressive in terminating idle compute nodes. One must set minimum size to the desired size in order to keep the fleet alive for long durations.
- The most efficient pipeline is to download subsets of data from S3 to each compute node's `/scratch`, process the subset, then upload results to S3, then delete the data and results. The granularity of this pipeline should depend on the local storage size of the compute node. If the storage is very small, this should be done for every file.
Compared with writing simultaneously to the shared NFS, this pipeline avoids write contention by writing to local scratch space and avoids the latency of reading from shared NFS by reading from local scratch as well.
- The best cell should be self-containing (works by itself if the ipython notebook is restarted or cluster is rebooted.)

# Understanding cfnCluster #
https://github.com/awslabs/cfncluster/blob/master/docs/source/autoscaling.rst
http://cfncluster.readthedocs.io/en/latest/processes.html

# Install cfnCluster #

Setup an Admin node. it could be local machine or an aws EC2 instance.

Install cfncluster on ADMIN https://github.com/awslabs/cfncluster or
 `sudo pip install cfncluster`. Version is cfncluster-1.3.1 (as of 3/16/2017)

 Reference: https://cfncluster.readthedocs.io/en/latest/getting_started.html

# Create custom AMI for cfncluster nodes #

http://cfncluster.readthedocs.io/en/latest/ami_customization.html
Create EC2 instance using Community AMI
- 16.04 ami-751f5315
- 14.04 ami-40185420
- Full list of cfnCluster AMIs https://github.com/awslabs/cfncluster/blob/master/amis.txt

Installed packages CFN_AMI11 ami-62194a02, in Community AMIs now.

Run `cfncluster configure`
or
```cp /usr/local/lib/python2.7/dist-packages/cfncluster/examples/config to /home/yuncong/.cfncluster
http://cfncluster.readthedocs.io/en/latest/configuration.html
custom_ami - ami-XXXXXXX
base_os - ubuntu14.04
compute_instance_type - m4.4xlarge
master_instance_type - m4.2xlarge
ebs_settings - custom
master_root_volume_size - 30
compute_root_volume_size - 30
volume_size - 50
vpc_id - see instance description
master_subnet_id - see instance description
aws_access_key_id - use access key, not IAM
aws_secret_access_key
aws_region_name
key_name
```

Master node has no choice but be on-demand. Compute node can be spot.

Cluster name must satisfy regular expression pattern: ``[a-zA-Z][-a-zA-Z0-9]``
```
Output:"MasterPublicIP"="52.53.116.181"
Output:"MasterPrivateIP"="172.31.21.42"
Output:"GangliaPublicURL"="http://52.53.116.181/ganglia/"
Output:"GangliaPrivateURL"="http://172.31.21.42/ganglia/"
```

# Timing #

- Create Master node takes 10 minutes.
- Compute node 6 minutes.

Then access with
`ssh -i aws/YuncongKey.pem ubuntu@ec2-54-67-87-143.us-west-1.compute.amazonaws.com`
Must specify `custom_ami` or `base_os` otherwise you cannot SSH to either master or compute nodes.

Security Group: Must enable defaultVPC and "AllowSSH22"
Enable "Allow5000" for flask server and "Allow8888" for jupyter notebook

# Monitor Cluster #

- EC2 Console
- Autoscaling Group Console
- CloudFormation Console
- Ganglia

# Access Key #
Note that DO NOT put any file containing access key in github repo. Otherwise AWS will detect it and inactivate it automatically.

# Jupyter Notebook #
Access from browser `https://<master node ip>:8888`

# Custom Bootstrap Actions #
http://cfncluster.readthedocs.io/en/latest/pre_post_install.html
Must make the S3 script public readable, otherwise `cfncluster create` will return 403.

# S3 #
Use this in bucket policy to enable make public by default.
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "MakeItPublic",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::mousebrainatlas-data/*"
        }
    ]
}

# Build Customized AMI #
http://cfncluster.readthedocs.io/en/latest/ami_customization.html

Must use a standalone instance. Should not use cluster's master or compute node.

The base CfnCluster AMI is often updated with new releases. This AMI has all of the components required for CfnCluster to function installed and configured. If you wish to customize an AMI for CfnCluster, you must start with this as the base.

Find the AMI which corresponds with the region you will be utilizing in the list here: https://github.com/awslabs/cfncluster/blob/master/amis.txt.
Within the EC2 Console, choose "Launch Instance".
Navigate to "Community AMIs", and enter the AMI id for your region into the search box.
Select the AMI, choose your instance type and properties, and launch your instance.
Log into your instance using the ec2-user and your SSH key.
Customize your instance as required
Run the following command to prepare your instance for AMI creation:
`sudo /usr/local/sbin/ami_cleanup.sh`
Stop the instance
Create a new AMI from the instance
Enter the AMI id in the custom_ami field within your cluster configuration.

# Expand Shared EBS #
http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-expand-volume.html#recognize-expanded-volume-linux
Stop instance.
Modify volume in console.
Restart instance.
`lsblk` shows new size but `df -h` still shows old size. Do `sudo resize2fs /dev/xvda1`.

# NFS #
Use master instance type - m4.2xlarge, larger memory on NFS server for a performance improvement (Runtime measured for Align step of Global Align)
Instance Type	Runtime
t2.micro	2687 seconds
m4.2xlarge	671 seconds
Set async option for NFS
Edit `/etc/exports`, change sync to async
Restart NFS server  `sudo service nfs-kernel-server restart `

# Ganglia #

`sudo apt-get install libapache2-mod-php7.0 php7.0-xml ; sudo /etc/init.d/apache2 restart`
Reference: http://blog.vuksan.com/2016/05/03/ganglia-webfrontend-ubuntu-1604-install-issue

# Sun Grid Engine #

Beginner Tutorial
* http://bioinformatics.mdc-berlin.de/intro2UnixandSGE/sun_grid_engine_for_beginners/README.html
* http://www.softpanorama.org/HPC/Grid_engine/Reference/sge_cheat_sheet.shtml#commonly_used_commands
* How to control number of hosts and slots http://stackoverflow.com/a/9018445/1550391

## Add user ubuntu to list of all grid managers ##
Change to super user `sudo -i`
Set environment variable `$SGE_ROOT`: `export SGE_ROOT=/opt/sge`
Add user: `/opt/sge/bin/lx-amd64/qconf -am ubuntu`
## Another simple way to enable admin permission ##
`sudo -u sgeadmin -i qconf -de ip-XXXXX.compute.internal`

`alias sudosgeadmin="sudo -u sgeadmin -i"`

## Remove execution host from gridengine ##
- first, you need to disable the host from queue to avoid any jobs to be allocated to this host
`qmod -d all.q@thishost.com`
- wait for jobs to be finished execution on this host, then kill the execution script
`qconf -ke thishost.com`
- remove it from the cluster, this opens an editor, just remove the lines referring to this host
`qconf -mq all.q`
- remove it from allhosts group, this also opens an editor, remove lines referring to this host
`qconf -mhgrp @allhosts`
- remove it from execution host list
`qconf -de thishost`
- I normally go to the host and delete the sge scripts as well

If still stuck deleting a host, grep hostnames in `/opt/sge/default/spool` and remove the strings.

Reference: https://resbook.wordpress.com/2011/03/21/remove-execution-host-from-gridengine/

## Performance Tuning ##

- Set minimum memory requirement to allow scheduling on a node as 5 GB: `qconf -mc` Change 0 under mem_free to 5G
- Change SGE schedule interval: `qconf -msconf` Change schedule_interval to 0:0:15

## Parallel environment ##
`qsub -pe mpi %(jobs_per_node)d -V -l mem_free=60G -o %(stdout_log)s -e %(stderr_log)s %(script)s`

# Startup Script #

https://ucsd-mousebrainatlas-scripts.s3.amazonaws.com/set_env.sh
```#!/bin/bash
echo "export RAW_DATA_DIR='/shared/data/CSHL_data'
export DATA_DIR='/shared/data/CSHL_data_processed'
export VOLUME_ROOTDIR='/shared/data/CSHL_volumes2'
export SCOREMAP_VIZ_ROOTDIR='/shared/data/CSHL_lossless_scoremaps_Sat16ClassFinetuned_v2'
export SVM_ROOTDIR='/shared/data/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers/'
export PATCH_FEATURES_ROOTDIR='/shared/data/CSHL_patch_features_Sat16ClassFinetuned_v2'
export SPARSE_SCORES_ROOTDIR='/shared/data/CSHL_patch_Sat16ClassFinetuned_v2_predictions'
export SCOREMAPS_ROOTDIR='/shared/data/CSHL_lossless_scoremaps_Sat16ClassFinetuned_v2'
export HESSIAN_ROOTDIR='/shared/data/CSHL_hessians/'
export REPO_DIR='/shared/MouseBrainAtlas'
export LABELING_DIR='/shared/CSHL_data_labelings_losslessAlignCropped'" >> /home/ubuntu/.bashrc
```

# GPU instance #

Create custom GPU instance AMI for cfncluster nodes
- Launch an EC2 instance using AMI from https://github.com/awslabs/cfncluster/blob/master/amis.txt (AMI chosen as base - ami-40185420) of type "g2.8xlarge" for GPU functionality
- Install driver nvidia-367(other drivers haven't been tested)
  - sudo apt-get install nvidia-367
  - nvidia-smi #to sanity check
- Install MXNet following official documentation (http://mxnet.io/get_started/ubuntu_setup.html)
- Prepare instance for AMI creation
  - sudo /usr/local/sbin/ami_cleanup.sh
- Stop the instance and create AMI
