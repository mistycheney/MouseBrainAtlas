#! /usr/bin/env python

import os
import shutil
import sys

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *

stack = sys.argv[1]
# stack = 'MD589'

first_sec, last_sec = section_range_lookup[stack]

# new_dir = sys.argv[1]
new_dir = '/home/yuncong/CSHL_data_labelingsViz/'+stack
if not os.path.exists(new_dir):
	os.makedirs(new_dir)

for sec in range(first_sec, last_sec+1):

	dm = DataManager(data_dir=os.environ['DATA_DIR'], 
			         repo_dir=os.environ['REPO_DIR'], 
			         result_dir=os.environ['RESULT_DIR'], 
			         labeling_dir=os.environ['LABELING_DIR'],
			         stack=stack, section=sec, load_mask=False)
	try:
		path, usr, ts = dm.load_review_result_path(username=None, timestamp='latest', suffix='consolidated')
		preview_img_path = path[:-4] + '.jpg'
		print preview_img_path
		shutil.copy(preview_img_path, new_dir)
	except:
		sys.stderr.write('error at section %d\n'%sec)

	