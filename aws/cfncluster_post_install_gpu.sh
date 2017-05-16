#! /bin/bash

sudo apt-get update

sudo apt-get install -y wget
wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py

sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install -y nvidia-367
sudo apt-get install mesa-common-dev
sudo apt-get install freeglut3-dev
sudo nvidia-xconfig --cool-bits=28

# Install all Python packages
sudo pip install numpy scipy matplotlib tables scikit-learn scikit-image multiprocess jupyter bloscpack pandas flask paramiko shapely boto3 opencv-python

# Install elastix
sudo apt-get install -y elastix

# Install imagemagick
sudo apt-get install -y imagemagick

# Code repo
REPO_DIR="/shared/MouseBrainAtlas"
git clone https://github.com/mistycheney/MouseBrainAtlas.git $REPO_DIR
chown -R ubuntu $REPO_DIR

# Kakadu
wget http://kakadusoftware.com/wp-content/uploads/2014/06/KDU79_Demo_Apps_for_Linux-x86-64_170108.zip
unzip KDU79_Demo_Apps_for_Linux-x86-64_170108.zip -d /home/ubuntu/

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
EOF
    chown -R ubuntu $JUPYTER_CONFIG_DIR
fi

# Install CUDA
# https://gist.github.com/albertstartup/9619faab6a2f6afdf4dc13f897d48a05
# https://gist.github.com/albertstartup/fed638a5d6862c9f0e8ffe8c3a74dbc8
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda
echo -e "export CUDA_HOME=/usr/local/cuda\nexport PATH=\$PATH:\$CUDA_HOME/bin\nexport LD_LIBRARY_PATH=\$LD_LINKER_PATH:\$CUDA_HOME/lib64" >> ~/.bashrc

# Install CuDNN
wget https://s3-us-west-1.amazonaws.com/mousebrainatlas-scripts/cudnn-8.0-linux-x64-v6.0.tgz
tar xf cudnn-8.0-linux-x64-v6.0.tgz
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/cudnn.h /usr/local/cuda/include/

# Install Mxnet
# http://mxnet.io/get_started/install.html
sudo pip install mxnet-cu80

echo """
export REPO_DIR=/shared/MouseBrainAtlas
alias sudosgeadmin=\"sudo -u sgeadmin -i\"
start_notebook() {REPO_DIR=$1 jupyter notebook --notebook-dir $1; }
alias increase_ebs_size=\"sudo resize2fs /dev/xvdb\"
""" >> /home/ubuntu/.bashrc
