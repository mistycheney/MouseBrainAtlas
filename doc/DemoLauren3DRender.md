```
cd /home/yuncong/Brain/ # or the repo dir you chose
source demo/set_env_variables.sh
python demo/download_render_demo_data.py
ENABLE_UPLOAD_S3=0 ENABLE_DOWNLOAD_S3=0 python demo/demo_vtk_render_atlas.py --experiments_config demo/lauren_experiments.csv 
```

The file `demo/lauren_experiments.csv` specifies the set of brains to display and the color of each. One can select which brains to show by changing the csv file.

In the 3D viewer, use mouse wheel to zoom and SHIFT+drag to move.
