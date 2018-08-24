```
git clone https://github.com/mistycheney/MouseBrainAtlas.git
cd MouseBrainAtlas/
source demo/set_env_variables.sh
python demo/donwload_render_demo_data.py
ENABLE_UPLOAD_S3=0 ENABLE_DOWNLOAD_S3=0 python demo/demo_vtk_render_atlas.py --experiments_config demo/lauren_experiments.csv 
```
