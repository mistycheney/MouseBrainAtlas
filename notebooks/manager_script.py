# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext autoreload
%autoreload 2

import os, json
from IPython.display import FileLink, Image, FileLinks
from pprint import pprint

import manager_utilities

# <markdowncell>

# You can read the details of a specific parameter setting. 0 is the id for default parameter setting.

# <codecell>

manager_utilities.load_param(91)

# <markdowncell>

# You can also tweak the parameters, and save to a different param_id

# <codecell>

# manager_utilities.create_param(88, n_superpixels=100)
# manager_utilities.create_param(89, n_superpixels=500)
# manager_utilities.create_param(90, n_superpixels=1000)
manager_utilities.create_param(91, n_superpixels=300)

# <markdowncell>

# Specify the image path, and parameter file.
# Then run the pipeline executable.

# <codecell>

cache_dir = 'scratch'
# img_dir = '~/ParthaData/PMD1305_reduce0_region0'
img_dir = 'data/dataset0'
img_name_full = 'PMD1305_244_reduce4_region0.tif'

img_path = os.path.join(img_dir, img_name_full)
img_name, ext = os.path.splitext(img_name_full)

param_id = 91
param_file = 'params/param%d.json'%param_id

# <codecell>

%%time

%run CrossValidationPipelineScriptShell.py {param_file} {img_path} -c {cache_dir}

# <markdowncell>

# View or download the results here

# <codecell>

FileLinks(os.path.join('scratch', img_name))

# <codecell>

# !rm scratch/{img_name}_param{param_id}_*.tif
# !convert scratch/{img_name}_param{param_id}_*.tif all.tif

