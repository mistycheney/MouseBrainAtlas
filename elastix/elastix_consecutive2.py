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

def parameter_file_to_dict(filename):
	d = {}
	with open(filename, 'r') as f:
		for line in f.readlines():
			if line.startswith('('):
				tokens = line[1:-2].split(' ')
				key = tokens[0]
				if len(tokens) > 2:
					value = []
					for v in tokens[1:]:
						try:
							value.append(float(v))
						except ValueError:
							value.append(v)
				else:
					v = tokens[1]
					try:
						value = (float(v))
					except ValueError:
						value = v
				d[key] = value

		return d

def parse_parameter_file(filepath):

	d = parameter_file_to_dict(filepath)

	rot_rad, x_mm, y_mm = d['TransformParameters']
	center = np.array(d['CenterOfRotationPoint']) / np.array(d['Spacing'])
	# center[1] = d['Size'][1] - center[1]

	xshift = x_mm / d['Spacing'][0]
	yshift = y_mm / d['Spacing'][1]

	R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
	              [np.sin(rot_rad), np.cos(rot_rad)]])
	shift = center + (xshift, yshift) - np.dot(R, center)
	T = np.vstack([np.column_stack([R, shift]), [0,0,1]])

	return T

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
first_sec = int(sys.argv[2])
last_sec = int(sys.argv[3])
suffix = 'thumbnail'

DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data'

prefix = stack + '_' + suffix

final_tranf_filename = os.path.join(DATA_DIR, prefix + '_finalTransfParams.pkl')
output_dir = create_if_not_exists(os.path.join('/tmp', prefix + '_output'))


transformation_to_previous_sec = {}
for moving_secind in range(first_sec+1, last_sec+1):
    param_fn = os.path.join(output_dir, 'output%dto%d/TransformParameters.0.txt'%(moving_secind, moving_secind-1))
    transformation_to_previous_sec[moving_secind] = parse_parameter_file(param_fn)

# with open(consecutive_transf_filename, 'w') as f:
# 	pickle.dump(transformation_to_previous_sec, f)s

transformation_to_first_sec = {}

for moving_secind in range(first_sec, last_sec+1):
    T_composed = np.eye(3)
    for i in range(first_sec+1, moving_secind+1):
        T_composed = np.dot(transformation_to_previous_sec[i], T_composed)
    transformation_to_first_sec[moving_secind] = T_composed

with open(final_tranf_filename, 'w') as f:
    pickle.dump(transformation_to_first_sec, f)