#! /usr/bin/env python

import os, sys
import numpy as np
from skimage.io import imread, imsave
import cPickle as pickle
from joblib import Parallel, delayed

stack = sys.argv[1]
input_dir = sys.argv[2]
DATA_DIR = '/home/yuncong/csd395/CSHL_data'

# tn_fns = [fn for fn in os.listdir(input_dir) if 'thumbnail' in fn]

# for fn in os.listdir(input_dir):
#     if 'thumbnail' in fn:

def get_bg_color(fn):

    im = imread(os.path.join(input_dir, fn))

    secind = int(fn[:-4].split('_')[1])

    c1 = np.bincount(im[...,0].flatten(), minlength=256)
    c1[255] = 0
    bg_r = np.argmax(c1)

    c2 = np.bincount(im[...,1].flatten(), minlength=256)
    c2[255] = 0
    bg_g = np.argmax(c2)

    c3 = np.bincount(im[...,2].flatten(), minlength=256)
    c3[255] = 0
    bg_b = np.argmax(c3)

    return (secind, (bg_r, bg_g, bg_b))

bg_map = dict(Parallel(n_jobs=16)(delayed(get_bg_color)(fn) for fn in os.listdir(input_dir) if 'thumbnail' in fn) )

tmp_dir = DATA_DIR + '/tmp'
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
with open(os.path.join(tmp_dir, stack + '_bgColor.pkl'), 'w') as f:
    pickle.dump(bg_map, f)