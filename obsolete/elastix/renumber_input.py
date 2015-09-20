#!/usr/bin/env python

import sys
import os
import shutil
import numpy as np

stack = sys.argv[1]
output_dir = sys.argv[2]
# output_dir = stack + '_renumbered'

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

filenames = os.listdir(stack)

secind_filename_map = dict([(int(fn.split('_')[-1][:-4]), fn) for fn in filenames])

min_secind = np.min(secind_filename_map.keys()) 

for secind, filename in secind_filename_map.iteritems():
	new_secind = secind - min_secind
	print filename + '->' + str(new_secind)
	shutil.copyfile(stack + '/' + filename, output_dir + '/' + str(new_secind) + '.png')

# os.chdir(output_dir)
# os.system('mogrify -format tif *.png')
