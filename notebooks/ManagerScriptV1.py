# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

import os
from IPython.display import FileLink, Image, FileLinks
import itertools
import manager_utilities

# <markdowncell>

# Specify the image path, and parameter file.

# <codecell>

output_dir = '../output'
data_dir = '../data'
param_dump_dir = '../params'

dataset_name = 'PMD1305_reduce2_region0'
img_idxs = [244,245]
param_ids = [2,3]

# <markdowncell>

# [This spreadsheet](https://docs.google.com/spreadsheets/d/1S189da_CxzC3GKISG3hZDG0n7mMycC0v4zTiRJraEUE/edit#gid=0) specifies the set of parameter values to test.

# <codecell>

parameters = manager_utilities.load_parameters('params.csv', redownload=False, dump_dir=param_dump_dir)

# <markdowncell>

# Then run the pipeline executable.

# <codecell>

%%time

for img_idx, param_id in itertools.product(img_idxs, param_ids):
    img_path = os.path.join(data_dir, dataset_name, dataset_name + '_%04d.tif'%img_idx)
    param_file = os.path.join(param_dump_dir, 'param%d.json'%param_id)
    
    %run CrossValidationPipelineScriptShellNoMagicV1.py {param_file} {img_path} {output_dir} 
    print '\n'

# <codecell>

Image(manager_utilities.get_img_filename('segmentation', img_name, param_id, cache_dir=cache_dir, ext='png'))

# <codecell>

Image(manager_utilities.get_img_filename('segmentation', img_name, param_id, cache_dir=cache_dir, ext='png'))

