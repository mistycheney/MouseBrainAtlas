#! /bin/bash

# Set aws credentials for AWS CLI
echo "export AWS_ACCESS_KEY_ID=$2" >> /home/ubuntu/.bashrc
echo "export AWS_SECRET_ACCESS_KEY=$3" >> /home/ubuntu/.bashrc
echo "export AWS_DEFAULT_REGION=$4" >> /home/ubuntu/.bashrc

sudo apt-get update

# Install all Python packages
sudo pip install --upgrade pip
sudo pip install numpy scipy matplotlib tables scikit-learn scikit-image multiprocess jupyter bloscpack pandas shapely boto3 opencv-python

# Install other utility programs
sudo apt-get install -y tree screen

# Ganglia
sudo apt-get install -y libapache2-mod-php7.0 php7.0-xml
sudo /etc/init.d/apache2 restart

# Setup Jupyter Notebook Server
CERTIFICATE_DIR="/home/ubuntu/jupyter_notebook_certificate"
JUPYTER_CONFIG_DIR="/home/ubuntu/.jupyter"

if [ ! -d "$CERTIFICATE_DIR" ]; then
    mkdir $CERTIFICATE_DIR
    openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout "$CERTIFICATE_DIR/mykey.key" -out "$CERTIFICATE_DIR/mycert.pem" -batch
    chown -R ubuntu $CERTIFICATE_DIR
fi

if [ ! -f "$JUPYTER_CONFIG_DIR/jupyter_notebook_config.py" ]; then
    # generate default config file
    #jupyter notebook --generate-config
    mkdir $JUPYTER_CONFIG_DIR

    # append notebook server settings
    cat <<EOF >> "$JUPYTER_CONFIG_DIR/jupyter_notebook_config.py"
# Set options for certfile, ip, password, and toggle off browser auto-opening
c.NotebookApp.certfile = u'$CERTIFICATE_DIR/mycert.pem'
c.NotebookApp.keyfile = u'$CERTIFICATE_DIR/mykey.key'
# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = '*'
c.NotebookApp.password = u'sha1:afce88b058a7:4c5afcebb62383f6f26404a08d6f5e89651709cb'
c.NotebookApp.open_browser = False

# It is a good idea to set a known, fixed port for server access
c.NotebookApp.port = 8888
c.NotebookApp.iopub_data_rate_limit = 10000000
EOF
    chown -R ubuntu $JUPYTER_CONFIG_DIR
fi

jupyter nbextension enable --py widgetsnbextension

# Set alias for getting sudo SGE privilege
echo "sudosgeadmin() { sudo -u sgeadmin -i \$1; }" >> /home/ubuntu/.bashrc

# Set an alias for updating the recognized disk size
# after manually changing it in the aws web console.
echo "increase_ebs_size() { sudo resize2fs /dev/xvdb; }" >> /home/ubuntu/.bashrc

# Set an alias for starting the notebook.
echo "start_notebook() { jupyter notebook --notebook-dir \$1 & }" >> /home/ubuntu/.bashrc

##################################################

# Clone code repo for Xiang
REPO_DIR_XIANG="/shared/MouseBrainAtlasXiang"
git clone https://github.com/xiangjiph/MouseBrainAtlas.git $REPO_DIR_XIANG
chown -R ubuntu $REPO_DIR_XIANG

# Set environment variable.
echo "export REPO_DIR=$REPO_DIR_XIANG" >> /home/ubuntu/.bashrc