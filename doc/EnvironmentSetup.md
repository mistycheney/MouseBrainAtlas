
# Software installation

`$ pip install activeatlas`

This will download the scripts and the package containing the reference anatomical model and the trained texture classifiers.

Edit `global_setting.py` to specify local data paths.

# Using iPython Notebook Server

If using AWS, 
- On your local machine, run:
`cfncluster --config <config> create <clusterName>`.
Wait until the cluster creation finishes to see the master node IP, or log onto AWS EC2 console to get the IP.
- Log in the master node. `ssh -i /home/yuncong/aws/YuncongKey.pem ubuntu@<server_ip>`.

If using the lab workstation,
- Log in workstation, `ssh <workstaton_ip>`.

The following steps work the same for both AWS and the lab workstation.
- Run `screen` to open a screen session (so the processes continue even if the terminal/SSH connection is closed)
- Run `jupyter notebook <project_repo_dir> &` to start a Jupyter notebook in the background.
- Run `screen -d` to detach the screen session.

Then on your local machine,
- Open a browser and go to `https//<server_ip>:8888` (assuming the Jupyter notebook uses port 8888). You can now access the notebook.


