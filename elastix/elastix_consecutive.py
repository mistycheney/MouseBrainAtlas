#!/usr/bin/env python

from subprocess import check_output, call
import os 
import numpy as np
import sys
from skimage.io import imread, imsave
import time
import re
import cPickle as pickle
from skimage.transform import warp, AffineTransform

def create_if_not_exists(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return path

def execute_command(cmd):
	print cmd

	try:
		retcode = call(cmd, shell=True)
		if retcode < 0:
			print >>sys.stderr, "Child was terminated by signal", -retcode
		else:
			print >>sys.stderr, "Child returned", retcode
	except OSError as e:
		print >>sys.stderr, "Execution failed:", e
		raise e


stack = sys.argv[1]
moving_secind = int(sys.argv[2])

suffix = 'thumbnail'

DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data'

prefix = stack + '_' + suffix

im_dir = os.path.join(DATA_DIR, prefix + '_padded')
output_dir = create_if_not_exists(os.path.join('/tmp', prefix + '_output'))
consecutive_transf_filename = os.path.join(DATA_DIR, prefix + '_consecTransfParams.pkl')

n_sections = len(os.listdir(im_dir))

rg_param = os.environ['GORDON_REPO_DIR'] + "/elastix/parameters/Parameters_Rigid.txt"

d = {'elastix_bin': os.environ['GORDON_ELASTIX'], 'rg_param': rg_param}

ext = 'tif'

d['output_subdir'] = os.path.join(output_dir, 'output%dto%d'%(moving_secind, moving_secind-1))
d['fixed_fn'] = os.path.join(im_dir, stack+'_%04d'%(moving_secind-1)+'_'+suffix+'_padded.'+ext)
d['moving_fn'] = os.path.join(im_dir, stack+'_%04d'%(moving_secind)+'_'+suffix+'_padded.'+ext)

create_if_not_exists(d['output_subdir'])

execute_command('%(elastix_bin)s -f %(fixed_fn)s -m %(moving_fn)s -out %(output_subdir)s -p %(rg_param)s' % d)