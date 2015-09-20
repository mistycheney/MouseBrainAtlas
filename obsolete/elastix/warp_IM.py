#! /usr/bin/env python

import sys
import os
import numpy as np
import cPickle as pickle

from joblib import Parallel, delayed

stack = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
first_secind = int(sys.argv[4])
last_secind = int(sys.argv[5])
suffix = sys.argv[6]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

DATAPROC_DIR = '/home/yuncong/csd395/CSHL_data_processed'

all_files = dict(sorted([(int(img_fn[:-4].split('_')[1]), img_fn) for img_fn in os.listdir(input_dir) if suffix in img_fn]))

if suffix == 'thumbnail':
    scale_factor = 1
elif suffix == 'lossy':
    scale_factor = 16
elif suffix == 'lossless':
    scale_factor = 32

with open(os.path.join(DATAPROC_DIR, stack + '_finalTransfParams.pkl'), 'r') as f:
    Ts = pickle.load(f)
    
for secind in range(first_secind, last_secind+1):

    T2 = Ts[secind].copy()
    T2[:2,2] = Ts[secind][:2, 2] * scale_factor
    T = np.linalg.inv(T2)

    d = {'sx':T[0,0],
         'sy':T[1,1],
         'rx':T[1,0],
         'ry':T[0,1],
         'tx':T[0,2],
         'ty':T[1,2],
         'input_fn': os.path.join(input_dir, all_files[secind]),
         'output_fn': os.path.join(output_dir, all_files[secind][:-4] + '_warped.tif')
        }

    os.system("convert %(input_fn)s -virtual-pixel background +distort AffineProjection '%(sx)f,%(rx)f,%(ry)f,%(sy)f,%(tx)f,%(ty)f' -crop 2000x1500+0+0\! -flatten -compress lzw %(output_fn)s"%d)