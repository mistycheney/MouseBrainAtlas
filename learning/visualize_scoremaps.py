#! /usr/bin/env python

import cv2

import os
import sys
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *

from collections import defaultdict
import pandas as pd
from joblib import Parallel, delayed

import time

stack = sys.argv[1]
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])

##################################################

labels = ['BackG', '5N', '7n', '7N', '12N', 'Pn', 'VLL', 
          '6N', 'Amb', 'R', 'Tz', 'RtTg', 'LRt', 'LC', 'AP', 'sp5']

scoremaps_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremaps_lossless_svm'

scoremapViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremapViz_svm'
if not os.path.exists(scoremapViz_rootdir):
    os.makedirs(scoremapViz_rootdir)
    
downscale_factor = 8

def func(sec):
    
    global stack
    
    scoremapViz_dir = os.path.join(scoremapViz_rootdir, stack, '%04d'%sec)
    if not os.path.exists(scoremapViz_dir):
        os.makedirs(scoremapViz_dir)

    dm = DataManager(stack=stack, section=sec)
    dm._load_image(['rgb-jpg'])

    dataset = '%(stack)s_%(sec)04d_roi1' % {'stack': stack, 'sec': sec}

    for l in labels_to_detect:
        
        scoremap_bp_filepath = scoremaps_rootdir + '/%(stack)s/%(slice)04d/%(stack)s_%(slice)04d_roi1_denseScoreMapLossless_%(label)s.hdf' \
            % {'stack': stack, 'slice': sec, 'label': l}
            
        if not os.path.exists(scoremap_bp_filepath):
            sys.stderr.write('No scoremap for %s for section %d\n' % (l, sec))
            continue
    
#         t = time.time()
#         scoremap = bp.unpack_ndarray_file(scoremap_bp_filepath)   
        scoremap = load_hdf(scoremap_bp_filepath)
#         sys.stderr.write('load scoremap: %.2f seconds\n' % (time.time() - t))
                
        interpolation_xmin, interpolation_xmax, \
        interpolation_ymin, interpolation_ymax = np.loadtxt(os.path.join(scoremaps_rootdir, stack, '%04d'%sec,
                                                                         '%(dataset)s_denseScoreMapLossless_%(label)s_interpBox.txt' % \
                                        {'dataset': dataset, 'label': l})).astype(np.int)
                
        dense_score_map_lossless = np.zeros((dm.image_height, dm.image_width))
        dense_score_map_lossless[interpolation_ymin:interpolation_ymax+1,
                                interpolation_xmin:interpolation_xmax+1] = scoremap
        
        scoremap_viz = plt.cm.hot(dense_score_map_lossless[::downscale_factor, ::downscale_factor])
#         scoremap_viz = plt.cm.hot(scoremap[::downscale_factor, ::downscale_factor])
        
        viz = (.3 * img_as_ubyte(scoremap_viz[..., :3]) + .7 * dm.image_rgb_jpg[::downscale_factor, ::downscale_factor]).astype(np.uint8)
#         viz = alpha_blending(scoremap_viz[..., :3], dm.image_rgb_jpg[::downscale_factor, ::downscale_factor], .3, 1.)

        cv2.putText(viz, l, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, ((0,0,0)), 3)
        
        cv2.imwrite(os.path.join(scoremapViz_dir, dataset+'_scoremapViz_%s.jpg' % l), 
                    img_as_ubyte(viz[..., [2,1,0]]))        
        
        
_ = Parallel(n_jobs=16)(delayed(func)(sec) for sec in range(first_sec, last_sec+1))
