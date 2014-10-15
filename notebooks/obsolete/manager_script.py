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

# [This spreadsheet](https://docs.google.com/spreadsheets/d/1S189da_CxzC3GKISG3hZDG0n7mMycC0v4zTiRJraEUE/edit#gid=0) specifies the set of parameter values to test.

# <codecell>

import csv
import gspread
import getpass

username = "cyc3700@gmail.com"
password = getpass.getpass()

docid = "1S189da_CxzC3GKISG3hZDG0n7mMycC0v4zTiRJraEUE"

client = gspread.login(username, password)
spreadsheet = client.open_by_key(docid)
for i, worksheet in enumerate(spreadsheet.worksheets()):
    filename = '../data/params.csv'
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(worksheet.get_all_values())

# <codecell>

parameters = []
with open('../data/params.csv', 'r') as f:
    param_reader = csv.DictReader(f)
    for param in param_reader:
        for k in param.iterkeys():
            if param[k] != '':
                try:
                    param[k] = int(param[k])
                except ValueError:
                    param[k] = float(param[k])
        if param['param_id'] == 0:
            default_param = param
        else:
            for k, v in param.iteritems():
                if v == '':
                    param[k] = default_param[k]
        parameters.append(param)

# <markdowncell>

# Specify the image path, and parameter file.
# Then run the pipeline executable.

# <codecell>

cache_dir = 'scratch'
img_dir = '../data/PMD1305_reduce2_region0'
img_name_full = 'PMD1305_reduce2_region0_0244.tif'

img_path = os.path.join(img_dir, img_name_full)
img_name, ext = os.path.splitext(img_name_full)

# <codecell>

%%time

for param_id in [0]:
# param_id = 4
    param_file = '../params/param%d.json'%param_id
    param = [p for p in parameters if p['param_id'] == param_id][0]
    json.dump(param, open(param_file, 'w'))

    %run CrossValidationPipelineScriptShellNoMagic.py {param_file} {img_path} -c {cache_dir}

# <codecell>

from skimage.io._plugins import freeimage_plugin as fi
image = np.zeros((32, 256, 256), 'uint16')
fi.write_multipage(image, 'multipage.tif')

# <codecell>

import tifffile
multipage = manager_utilities.regulate_images([segmentation, textonmap, texton_saliency_map])
multipage = multipage.astype(np.uint16)
tifffile.imsave(os.path.join(args.cache_dir, 'multipage.tif'), multipage)

# <markdowncell>

# View or download the results here

# <codecell>

!rm scratch/{img_name}_param{param_id}_*.tif
!convert scratch/{img_name}_param{param_id}_*.tif all.tif

