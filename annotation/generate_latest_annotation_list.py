#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *

stack = sys.argv[1]
first_bs_sec, last_bs_sec = section_range_lookup[stack]

dm = DataManager(stack=stack, labeling_dir='/home/yuncong/csd395/CSHL_data_labelings_losslessAlignCropped')

#########################

username = 'yuncong'

f = open(os.path.join(dm.root_labelings_dir, stack + '_' + username + '_latestAnnotationFilenames.txt'), 'w')

for sec in range(first_bs_sec, last_bs_sec + 1):

    dm.set_slice(sec)
    
    ret = dm.load_review_result_path(username, 'latest', suffix='consolidated')
    if ret is not None:
        fn = ret[0]
        print fn
        f.write(fn + '\n')
        
f.close()

#########################

username = 'autoAnnotate'

f = open(os.path.join(dm.root_labelings_dir, stack + '_' + username + '_latestAnnotationFilenames.txt'), 'w')

for sec in range(first_bs_sec, last_bs_sec + 1):

    dm.set_slice(sec)
    
    ret = dm.load_review_result_path(username, 'latest', suffix='consolidated')
    if ret is not None:
        fn = ret[0]
        print fn
        f.write(fn + '\n')
        
f.close()