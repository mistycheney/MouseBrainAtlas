#! /usr/bin/env python

import sys
import os
import time

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg') # https://stackoverflow.com/a/3054314
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from registration_utilities import *
from annotation_utilities import *
from metadata import *
from data_manager import *

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')

parser.add_argument("brain_name", type=str, help="Brain name")
parser.add_argument("--tb_version", type=str, help="Image version (Default: None)")
parser.add_argument("--tb_resol", type=str, help="Image resolution (Default: %(default)s)", default='thumbnail')
parser.add_argument("--output_resol", type=str, help="Output volume resolution (Default: %(default)s)", default='10.0um')

args = parser.parse_args()

stack = args.brain_name

output_resolution = args.output_resol
tb_version = args.tb_version
tb_resol = args.tb_resol

images = {}

#     for sec in metadata_cache['valid_sections_all'][stack]:
for sec in metadata_cache['valid_sections'][stack]:

    img_rgb = DataManager.load_image_v2(stack, section=sec, 
                                        resol=tb_resol, 
                                        prep_id='alignedWithMargin', 
                                        version=tb_version)
    img = img_as_ubyte(rgb2gray(img_rgb)) # Use greylevel

    mask = DataManager.load_image_v2(stack=stack, section=sec, 
                                     prep_id='alignedWithMargin', 
                                     resol=tb_resol, 
                                     version='mask')
    img[~mask] = 0

    images[sec] = img

# Specify isotropic resolution of the output volume.
voxel_size_um = convert_resolution_string_to_um(resolution=output_resolution, stack=stack)

input_image_resolution_um = convert_resolution_string_to_um(resolution=tb_resol, stack=stack)

volume_outVolResol, volume_origin_wrt_wholebrainWithMargin_outVolResol = images_to_volume_v2(images=images, 
                                            spacing_um=20.,
                                            in_resol_um=input_image_resolution_um,
                                            out_resol_um=voxel_size_um)
print volume_outVolResol.shape

##############################################################

prep5_origin_wrt_prep1_tbResol = DataManager.load_cropbox_v2(stack=stack, only_2d=True, prep_id='alignedWithMargin')

loaded_cropbox_resol = 'thumbnail'

prep5_origin_wrt_prep1_outVolResol = prep5_origin_wrt_prep1_tbResol * \
convert_resolution_string_to_um(resolution=loaded_cropbox_resol, stack=stack) / voxel_size_um

wholebrainWithMargin_origin_wrt_wholebrain_outVolResol = np.r_[np.round(prep5_origin_wrt_prep1_outVolResol).astype(np.int)[[0,2]], 0]
# wholebrainWithMargin_origin_wrt_wholebrain = np.array([0,0,0])

volume_origin_wrt_wholebrain_outVolResol = volume_origin_wrt_wholebrainWithMargin_outVolResol + wholebrainWithMargin_origin_wrt_wholebrain_outVolResol

########################################

stack_spec = dict(name=stack,
                  resolution=output_resolution,
                  prep_id='wholebrainWithMargin',
                  vol_type='intensity')

save_data(volume_outVolResol, 
          fp=DataManager.get_original_volume_filepath_v2(stack_spec=stack_spec, structure=None))

save_data(volume_origin_wrt_wholebrain_outVolResol, 
          fp=DataManager.get_original_volume_origin_filepath_v3(stack_spec=stack_spec, structure=None))
