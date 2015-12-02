#! /usr/bin/env python

import sys
import os
import shutil
import numpy as np

stack = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
output_jp2_dir = sys.argv[4]
arg_file = sys.argv[5]
suffix = sys.argv[6]

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

if not os.path.exists(output_jp2_dir):
	os.makedirs(output_jp2_dir)

with open(arg_file, 'r') as f:
        arg_tuples = map(lambda x: x.split(' '), f.readlines())
        arg_tuples = [(x[0], int(x[1])) for x in arg_tuples]

d = {'input_dir': input_dir,
'output_dir': output_dir,
'output_jp2_dir': output_jp2_dir,
'suffix': suffix}

for fn, new_ind in arg_tuples:
        d['fn_base'] = fn[:-4]
        d['new_fn_base'] = stack + '_%04d'%new_ind

        if suffix == 'thumbnail':
                shutil.copy(input_dir + '/' + fn, output_dir + '/' + d['new_fn_base'] + '_thumbnail.tif')
        elif suffix == 'lossy' or suffix == 'lossless':
                os.system('kdu_expand_patched -i %(input_dir)s/%(fn_base)s_%(suffix)s.jp2 -o %(output_dir)s/%(new_fn_base)s_%(suffix)s.tif' % d)
                os.system('ln -s %(input_dir)s/%(fn_base)s_%(suffix)s.jp2 %(output_jp2_dir)s/%(new_fn_base)s_%(suffix)s.jp2' % d)

f.close()