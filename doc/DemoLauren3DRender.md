Yuncong has already done the installation. Lauren just need to follow instructions in "Regular use".

# Installation

```
＃ Install vtk 7.1.1 (Cannot use newest version 8 because the window crashes immediately after it launches)
cd /home/yuncong
wget https://www.vtk.org/files/release/7.1/vtkpython-7.1.1-Linux-64bit.tar.gz
tar xfz vtkpython-7.1.1-Linux-64bit.tar.gz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/vtkpython-7.1.1-Linux-64bit/lib/
export PYTHONPATH=$PYTHONPATH:`pwd`/vtkpython-7.1.1-Linux-64bit/lib/python2.7/site-packages/

# Setup python virtual environment
sudo pip install virtualenv
pip install --user virtualenvwrapper
export WORKON_HOME=~/virtualenvs
source ~/.local/bin/virtualenvwrapper.sh
mkdir -p $WORKON_HOME
mkvirtualenv mousebrainatlas

# Enter the repository folder (your path might be different from this)
cd /home/yuncong/Brain/

# Install required python packages
sudo apt-get install libgeos-dev
pip install cython==0.28.5 # see this issue https://github.com/h5py/h5py/issues/535
pip install -r requirements.txt
pip install git+https://github.com/pmneila/PyMCubes.git@9fd6059

```

# Regular use

First log into Ubuntu or switch account to yuncong (use the upper-right corner gear icon).

```
workon mousebrainatlas

cd /home/yuncong/Brain/
source demo/set_env_variables.sh
source demo/set_vtk_env.sh

python demo/download_demo_data_render3d.py
python demo/demo_render3d.py --experiments_config demo/lauren_experiments.csv 
```

In case of "X Error of failed request: ...", follow the fix [here](https://askubuntu.com/a/882047)

- The file `demo/lauren_experiments.csv` specifies which experiments to display markers for and the color of each.
- The file `render_config_atlas.csv` specifies the color/opacity of each atlas structure.
- In the 3D viewer, use mouse wheel to zoom and SHIFT+drag to move. Press Q to quit.

Run `deactivate` to exit the virtualenv.
