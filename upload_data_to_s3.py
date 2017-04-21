#! /usr/bin/env python

import sys, os
sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import DataManager
from metadata import *

# for stack in ['MD642', 'MD652', 'MD653', 'MD635']:
for stack in ['MD642']:
# for stack in ['MD590', 'MD591', 'MD592', 'MD593', 'MD595', 'MD598', 'MD599', 'MD602', 'MD603']:
    anchor_fn = DataManager.load_anchor_filename(stack)
    execute_command("aws s3 cp /home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_cropbox.txt s3://mousebrainatlas-data/CSHL_data_processed/%(stack)s/%(stack)s_cropbox.txt" % {'stack': stack})
    execute_command("aws s3 cp /home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_anchor.txt s3://mousebrainatlas-data/CSHL_data_processed/%(stack)s/%(stack)s_anchor.txt" % {'stack': stack})
    execute_command("aws s3 cp /home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_sorted_filenames.txt s3://mousebrainatlas-data/CSHL_data_processed/%(stack)s/%(stack)s_sorted_filenames.txt" % {'stack': stack})
    execute_command("aws s3 cp --recursive /home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_elastix_output/ s3://mousebrainatlas-data/CSHL_data_processed/%(stack)s/%(stack)s_elastix_output/" % {'stack': stack, 'anchor_fn': anchor_fn})
    execute_command("aws s3 cp --recursive /home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_thumbnail_unsorted_alignedTo_%(anchor_fn)s/ s3://mousebrainatlas-data/CSHL_data_processed/%(stack)s/%(stack)s_thumbnail_alignedTo_%(anchor_fn)s/" % {'stack': stack, 'anchor_fn': anchor_fn})
    execute_command("aws s3 cp --recursive /home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_thumbnail_unsorted_alignedTo_%(anchor_fn)s_cropped/ s3://mousebrainatlas-data/CSHL_data_processed/%(stack)s/%(stack)s_thumbnail_alignedTo_%(anchor_fn)s_cropped/" % {'stack': stack, 'anchor_fn': anchor_fn})
    execute_command("aws s3 cp --recursive /home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_lossless_alignedTo_%(anchor_fn)s_cropped/ s3://mousebrainatlas-data/CSHL_data_processed/%(stack)s/%(stack)s_lossless_alignedTo_%(anchor_fn)s_cropped/" % {'stack': stack, 'anchor_fn': anchor_fn})
    execute_command("aws s3 cp --recursive /home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_lossless_alignedTo_%(anchor_fn)s_cropped_compressed/ s3://mousebrainatlas-data/CSHL_data_processed/%(stack)s/%(stack)s_lossless_alignedTo_%(anchor_fn)s_cropped_compressed/" % {'stack': stack, 'anchor_fn': anchor_fn})
    execute_command("aws s3 cp --recursive /home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_submasks/ s3://mousebrainatlas-data/CSHL_data_processed/%(stack)s/%(stack)s_submasks/" % {'stack': stack})
    execute_command("aws s3 cp --recursive /home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_masks/ s3://mousebrainatlas-data/CSHL_data_processed/%(stack)s/%(stack)s_masks/" % {'stack': stack})
    execute_command("aws s3 cp --recursive /home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_masks_alignedTo_%(anchor_fn)s/ s3://mousebrainatlas-data/CSHL_data_processed/%(stack)s/%(stack)s_masks_alignedTo_%(anchor_fn)s/" % {'stack': stack, 'anchor_fn': anchor_fn})
    execute_command("aws s3 cp --recursive /home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_masks_alignedTo_%(anchor_fn)s_cropped/ s3://mousebrainatlas-data/CSHL_data_processed/%(stack)s/%(stack)s_masks_alignedTo_%(anchor_fn)s_cropped/" % {'stack': stack, 'anchor_fn': anchor_fn})

for stack in ['MD642']:
    execute_command("aws s3 cp --recursive /home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_custom_transforms/ s3://mousebrainatlas-data/CSHL_data_processed/%(stack)s/%(stack)s_custom_transforms/" % {'stack': stack, 'anchor_fn': anchor_fn})
