#! /usr/bin/env python

import sys
import os
import time

import numpy as np
from multiprocess import Pool

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from data_manager import *
from metadata import *
from cell_utilities import *

########################################

import argparse
import json

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Detect cells using cell profiler.')

parser.add_argument("stack", type=str, help="Stack")
parser.add_argument("filenames", type=str, help="filenames")
args = parser.parse_args()

#########################################

stack = args.stack
filenames = json.loads(args.filenames)

img_w, img_h = metadata_cache['image_shape'][stack]

def detect_cells(fn):

    if is_invalid(fn):
        return

    input_img_fp = get_cell_data_filepath(stack=stack, fn=fn, what='image_inverted', ext='tif')

    ##########################
    # Split image into tiles
    ##########################

    # sat_filename = DataManager.get_image_filepath(stack=stack, fn=fn, resol='lossless', version='cropped_gray')
    sat_filename = DataManager.get_image_filepath(stack=stack, fn=fn, resol='lossless', version='cropped_gray_contrast_stretched')
    download_from_s3(sat_filename)
    copyto_sat_filename = get_cell_data_filepath(stack=stack, fn=fn, what='image_gray', ext='tif')
    create_parent_dir_if_not_exists(copyto_sat_filename)
    execute_command("cp %s %s" % (sat_filename, copyto_sat_filename))
    execute_command('convert %(img_orig_fp)s -negate -auto-level %(img_fp)s' % dict(img_orig_fp=copyto_sat_filename,
                                                                   img_fp=input_img_fp))
    upload_to_s3(copyto_sat_filename)
    upload_to_s3(input_img_fp)

    tile_h, tile_w = (5000, 5000)
    i = 0
    for iy, y0 in enumerate(np.arange(0, img_h, 5000)):
        for ix, x0 in enumerate(np.arange(0, img_w, 5000)):
            tile_fp = get_cell_data_filepath(stack=stack, fn=fn, what='image_inverted_%02d'%i, ext='tif')
            execute_command("convert %(img_fp)s -crop %(tile_w)dx%(tile_h)d+%(x0)d+%(y0)d %(tile_fp)s" % \
                            dict(img_fp=input_img_fp, x0=x0, y0=y0, tile_w=tile_w, tile_h=tile_h, tile_fp=tile_fp))
            i += 1

    ##################################

    input_fps = [os.path.splitext(input_img_fp)[0] + '_%02d.tif' % tile_idx for tile_idx in range(12)]

    ###############
    # CellProfiler
    ###############

    with open('/tmp/cellprofiler_filelist_%(fn)s.txt' % {'fn':fn}, 'w') as f:
        for input_fp in input_fps:
            f.write(input_fp + '\n')

    t = time.time()
    execute_command(CELLPROFILER_EXEC + ' -c --file-list=/tmp/cellprofiler_filelist_%(fn)s.txt -p %(pipeline_fp)s' % \
                    dict(pipeline_fp=CELLPROFILER_PIPELINE_FP, fn=fn))
    sys.stderr.write('Cell profiler: %.2f seconds.\n' % (time.time()-t)) # 300s

    from scipy.io import loadmat

    labelmap_alltiles = []
    for input_fp in input_fps:
        prefix = os.path.splitext(input_fp)[0]
        labelmap = loadmat(prefix + '_labeled.mat')['Image']
        execute_command('rm ' + prefix + '_labeled.mat')
        labelmap_alltiles.append(labelmap)

    ###########################
    # Mosaic for cell profiler
    ###########################

    origins = []
    for iy, y0 in enumerate(np.arange(0, img_h, 5000)):
        for ix, x0 in enumerate(np.arange(0, img_w, 5000)):
            origins.append((x0, y0))

    alg = 'cellprofiler'

    big_labelmap = np.zeros((img_h, img_w), dtype=np.int64)
    n = 0
    for i, input_fp in enumerate(input_fps):
        prefix = os.path.splitext(input_fp)[0]
        labelmap = labelmap_alltiles[i].astype(np.int64) # astype(np.int64) is important, otherwise results in negative label values.
        x0, y0 = origins[i]
        big_labelmap[y0:y0+5000, x0:x0+5000][labelmap != 0] = labelmap[labelmap != 0] + n
        n += labelmap.max()

    labelmap_fp = os.path.splitext(input_img_fp)[0] + '_labelmap_%(alg)s.bp' % dict(alg=alg)
    bp.pack_ndarray_file(big_labelmap, labelmap_fp)
    upload_to_s3(labelmap_fp)
    
    for fp in input_fps:
        execute_command('rm ' + fp)        

t = time.time()

pool = Pool(NUM_CORES/2)
pool.map(detect_cells, filenames)
pool.close()
pool.join()

sys.stderr.write('Overall time: %.2f seconds.\n' % (time.time()-t))
