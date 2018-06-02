#! /usr/bin/env python

# This must be before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import sys
import os
import time

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from registration_utilities import *
from metadata import *
from data_manager import *

from multiprocess import Pool

stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])

DETECTED_CELLS_DIR = '/home/yuncong/csd395/CSHL_cells_v2/detected_cells'
img_w, img_h = metadata_cache['image_shape'][stack]

s2f = metadata_cache['sections_to_filenames'][stack]

def detect_cells(sec):

    img_fn = s2f[sec]

    if img_fn in ['Nonexisting', 'Rescan', 'Placeholder']:
        return

#     input_img_fp = DETECTED_CELLS_DIR + '/%(stack)s/%(img_fn)s/%(img_fn)s_image_inverted.jpg' % dict(stack=stack, img_fn=img_fn)
    input_img_fp = os.path.join(DETECTED_CELLS_DIR, stack, img_fn, img_fn + '_image_inverted.jpg')

    ##########################
    # Split image into tiles
    ##########################

    if not os.path.exists(os.path.splitext(input_img_fp)[0] + '_00.tif'):

        sat_filename = DataManager.get_image_filepath(stack=stack, section=sec, resol='lossless', version='saturation')
        copyto_sat_filename = DETECTED_CELLS_DIR + '/%(stack)s/%(fn)s/%(fn)s_image_saturation.jpg' % {'fn':img_fn, 'stack':stack}

        create_if_not_exists(os.path.dirname(copyto_sat_filename))

        execute_command("cp %s %s" % (sat_filename, copyto_sat_filename))

        execute_command('convert %(img_orig_fp)s -negate -auto-level %(img_fp)s' % dict(img_orig_fp=copyto_sat_filename,
                                                                       img_fp=input_img_fp))

        tile_h, tile_w = (5000, 5000)

        i = 0
        for iy, y0 in enumerate(np.arange(0, img_h, 5000)):
            for ix, x0 in enumerate(np.arange(0, img_w, 5000)):
                tile_fp = os.path.join(DETECTED_CELLS_DIR, stack, img_fn, img_fn + '_image_inverted_%02d.tif') % i
                execute_command("convert %(img_fp)s -crop %(tile_w)dx%(tile_h)d+%(x0)d+%(y0)d %(tile_fp)s" % \
                                dict(img_fp=input_img_fp, x0=x0, y0=y0, tile_w=tile_w, tile_h=tile_h, tile_fp=tile_fp))
                i += 1

    ##################################

    input_fps = [os.path.splitext(input_img_fp)[0] + '_%02d.tif' % tile_idx for tile_idx in range(12)]

    #############
    # Farsight
    #############

    output_fps = [os.path.splitext(input_fp)[0] + '_labeled_farsight.tif' for input_fp in input_fps]

    FARSIGHT_EXEC = '/home/yuncong/csd395/Farsight-0.4.4-Linux/bin/segment_nuclei'

    # # Use without parameters

    # for input_fp, output_fp in zip(input_fps, output_fps):
    #     execute_command(FARSIGHT_EXEC + ' %(input_fp)s %(output_fp)s' % \
    #                    dict(input_fp=input_fp,
    #                         output_fp=output_fp))

    # Use with parameters

    FARSIGHT_PARAMS_FP = '/home/yuncong/csd395/CSHL_cells_v2/farsight_parameters.txt'

    with open(FARSIGHT_PARAMS_FP, 'w') as f:
        f.write('! Segmentation parameters File\n! All parameters are case sensitive\n')
        f.write('min_object_size 100\n')

    for input_fp, output_fp in zip(input_fps, output_fps):

        execute_command(FARSIGHT_EXEC + ' %(input_fp)s %(output_fp)s %(params)s' % \
                   dict(input_fp=input_fp,
                        output_fp=output_fp,
                        params=FARSIGHT_PARAMS_FP))

    labelmap_alltiles = []
    for input_fp in input_fps:
        prefix = os.path.splitext(input_fp)[0]
        labelmap = imread(prefix + '_labeled_farsight.tif')
        labelmap_alltiles.append(labelmap)
#         bp.pack_ndarray_file(labelmap, prefix + '_labelmap_farsight.bp')
        execute_command('rm ' + prefix + '_seg_final.dat')
        execute_command('rm ' + prefix + '_seedPoints.txt')
        execute_command('rm ' + prefix + '_labeled_farsight.tif')

    ######################
    # Mosaic for farsight
    ######################

    origins = []
    for iy, y0 in enumerate(np.arange(0, img_h, 5000)):
        for ix, x0 in enumerate(np.arange(0, img_w, 5000)):
            origins.append((x0, y0))

    alg = 'farsight'

    big_labelmap = np.zeros((img_h, img_w), dtype=np.int64)

    n = 0

    for i, input_fp in enumerate(input_fps):
        prefix = os.path.splitext(input_fp)[0]
#         labelmap = bp.unpack_ndarray_file(prefix + '_labelmap_%(alg)s.bp' % dict(alg=alg)).astype(np.int64)
        labelmap = labelmap_alltiles[i]

        x0, y0 = origins[i]
        big_labelmap[y0:y0+5000, x0:x0+5000][labelmap != 0] = labelmap[labelmap != 0] + n
        n += labelmap.max()

    labelmap_fp = os.path.splitext(input_img_fp)[0] + '_labelmap_%(alg)s.bp' % dict(alg=alg)

    bp.pack_ndarray_file(big_labelmap, labelmap_fp)

#     for tile_i in range(12):
#         execute_command('rm %(DETECTED_CELLS_DIR)s/%(stack)s/%(img_fn)s/%(img_fn)s_image_inverted_%(tile_i)02d.tif' % \
#                         dict(DETECTED_CELLS_DIR=DETECTED_CELLS_DIR, stack=stack, img_fn=img_fn, tile_i=tile_i))
#         execute_command('rm %(DETECTED_CELLS_DIR)s/%(stack)s/%(img_fn)s/%(img_fn)s_image_inverted_%(tile_i)02d_labelmap_cellprofiler.bp' % \
#                         dict(DETECTED_CELLS_DIR=DETECTED_CELLS_DIR, stack=stack, img_fn=img_fn, tile_i=tile_i))

    # Generate labelmap viz
    t = time.time()

    viz = img_as_ubyte(label2rgb(big_labelmap, bg_label=0, bg_color=(0, 0, 0)))
    cv2.imwrite(os.path.splitext(input_img_fp)[0] + '_labelmap_%(alg)s.png' % dict(alg=alg), viz);

    sys.stderr.write('Generate labelmap viz: %.2f seconds.\n' % (time.time()-t)) # 60s

    ###############
    # CellProfiler
    ###############

    CELLPROFILER_EXEC = 'cellprofiler' # /usr/local/bin/cellprofiler
    CELLPROFILER_PIPELINE_FP = '/home/yuncong/csd395/CSHL_cells_v2/SegmentCells.cppipe'

    # import uuid
    # uid = uuid.uuid4()
    with open('/tmp/cellprofiler_filelist_%04d.txt' % sec, 'w') as f:
        for input_fp in input_fps:
            f.write(input_fp + '\n')

    t = time.time()

    execute_command(CELLPROFILER_EXEC + ' -c --file-list=/tmp/cellprofiler_filelist_%(sec)04d.txt -p %(pipeline_fp)s' % \
                    dict(pipeline_fp=CELLPROFILER_PIPELINE_FP, sec=sec))

    sys.stderr.write('Cell profiler: %.2f seconds.\n' % (time.time()-t)) # 300s

    from scipy.io import loadmat

    labelmap_alltiles = []
    for input_fp in input_fps:
        prefix = os.path.splitext(input_fp)[0]
        labelmap = loadmat(prefix + '_labeled.mat')['Image']
#         bp.pack_ndarray_file(labelmap, prefix + '_labelmap_cellprofiler.bp')
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
#         labelmap = bp.unpack_ndarray_file(prefix + '_labelmap_%(alg)s.bp' % dict(alg=alg)).astype(np.int64)
        labelmap = labelmap_alltiles[i]

        x0, y0 = origins[i]
        big_labelmap[y0:y0+5000, x0:x0+5000][labelmap != 0] = labelmap[labelmap != 0] + n
        n += labelmap.max()

    labelmap_fp = os.path.splitext(input_img_fp)[0] + '_labelmap_%(alg)s.bp' % dict(alg=alg)

    bp.pack_ndarray_file(big_labelmap, labelmap_fp)

#     for tile_i in range(12):
#         execute_command('rm %(DETECTED_CELLS_DIR)s/%(stack)s/%(img_fn)s/%(img_fn)s_image_inverted_%(tile_i)02d.tif' % \
#                         dict(DETECTED_CELLS_DIR=DETECTED_CELLS_DIR, stack=stack, img_fn=img_fn, tile_i=tile_i))
#         execute_command('rm %(DETECTED_CELLS_DIR)s/%(stack)s/%(img_fn)s/%(img_fn)s_image_inverted_%(tile_i)02d_labelmap_cellprofiler.bp' % \
#                         dict(DETECTED_CELLS_DIR=DETECTED_CELLS_DIR, stack=stack, img_fn=img_fn, tile_i=tile_i))

    # Generate labelmap viz
    t = time.time()

    viz = img_as_ubyte(label2rgb(big_labelmap, bg_label=0, bg_color=(0, 0, 0)))
    cv2.imwrite(os.path.splitext(input_img_fp)[0] + '_labelmap_%(alg)s.png' % dict(alg=alg), viz);

    sys.stderr.write('Generate labelmap viz: %.2f seconds.\n' % (time.time()-t)) # 60s


t = time.time()

pool = Pool(12)
pool.map(detect_cells, range(first_sec, last_sec+1))
pool.close()
pool.join()

sys.stderr.write('Overall time: %.2f seconds.\n' % (time.time()-t))
