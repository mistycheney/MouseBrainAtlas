```
cd /home/yuncong/Brain/ # or the repo dir you chose

# Install required python packages
sudo apt-get install libgeos-dev
pip install -r requirements.txt

＃ Install vtk 7.1.1 (With newest version 8 the window crashes immediately after it launches)
wget https://www.vtk.org/files/release/7.1/vtkpython-7.1.1-Linux-64bit.tar.gz
tar xfz vtkpython-7.1.1-Linux-64bit.tar.gz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/vtkpython-7.1.1-Linux-64bit/lib/
export PYTHONPATH=$PYTHONPATH:`pwd`/vtkpython-7.1.1-Linux-64bit/lib/python2.7/site-packages/

source demo/set_env_variables.sh
python demo/download_render_demo_data.py
ENABLE_UPLOAD_S3=0 ENABLE_DOWNLOAD_S3=0 python demo/demo_vtk_render_atlas.py --experiments_config demo/lauren_experiments.csv 
```

The file `demo/lauren_experiments.csv` specifies the set of brains to display and the color of each. One can select which brains to show by changing the csv file.

In the 3D viewer, use mouse wheel to zoom and SHIFT+drag to move.
