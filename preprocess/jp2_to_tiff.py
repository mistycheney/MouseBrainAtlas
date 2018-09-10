#! /usr/bin/env python

import os
import sys
import time
import argparse

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Rename thumbnail images according to our naming format with consecutive section numbers')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("spec", type=str, help="Path to input files specification as json file")
args = parser.parse_args()

stack = args.stack_name
with open(args.spec, 'r') as f:
    input_spec = json.load(f)

data_dirs = {}
imageName_to_filepath_mapping = {}
filepath_to_imageName_mapping = {}
for spec_one_version in input_spec:
    vr = (spec_one_version['version'], spec_one_version['resolution'])
    data_dirs[vr] = spec_one_version['data_dirs']
    filepath_to_imageName_mapping[vr] = spec_one_version['filepath_to_imageName_mapping']
    imageName_to_filepath_mapping[vr] = spec_one_version['imageName_to_filepath_mapping']

image_names_all_data_dirs_flattened = set([])
image_names_all_data_dirs = {}
for vr, data_dir in data_dirs.iteritems():
    if data_dir is None: continue
    image_names = set([])
    if vr in filepath_to_imageName_mapping:
        for fn in os.listdir(data_dir):
            g = re.search(filepath_to_imageName_mapping[vr], os.path.join(data_dir, fn))
            if g is not None:
                img_name = g.groups()[0]
                image_names.add(img_name)
                image_names_all_data_dirs_flattened.add(img_name)
    image_names_all_data_dirs[vr] = image_names
    
print "Found %d images.\n" % len(image_names_all_data_dirs_flattened)

# Make sure the every image has all three channels.
for vr, img_names in image_names_all_data_dirs.iteritems():
    print vr, 'missing:'
    print image_names_all_data_dirs_flattened - img_names

create_if_not_exists(DataManager.get_image_dir_v2(stack=stack, prep_id=None, resol='raw'))

# The KDU program automatically uses all cores, so we just set jobs_per_node = 1.
run_distributed('export LD_LIBRARY_PATH=%(kdu_dir)s:$LD_LIBRARY_PATH; %(kdu_bin)s -i \"%%(in_fp)s\" -o \"%%(out_fp)s\"' % \
                {'kdu_bin': KDU_EXPAND_BIN, 'kdu_dir': os.path.dirname(KDU_EXPAND_BIN)},
                kwargs_list={'in_fp': [imageName_to_filepath_mapping[(None, 'raw')] % img_name
                                       for img_name in list(image_names_all_data_dirs_flattened)], 
                             'out_fp': [DataManager.get_image_filepath_v2(stack=stack, prep_id=None, 
                                        resol='raw', version=None, fn=img_name) 
                                        for img_name in list(image_names_all_data_dirs_flattened)]},
                argument_type='single',
                jobs_per_node=1,
                local_only=True)
