#!/usr/bin/env python

import os 
import numpy as np
import sys
import cPickle as pickle

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

stack = sys.argv[1]
input_dir = sys.argv[2]
first_sec = int(sys.argv[3])
last_sec = int(sys.argv[4])
anchor_sec = int(sys.argv[5])

# DATAPROC_DIR = '/home/yuncong/csd395/CSHL_data_processed'
DATAPROC_DIR = os.environ['DATA_DIR']

final_tranf_filename = os.path.join(DATAPROC_DIR, stack + '_finalTransfParams.pkl')

transformation_to_previous_sec = {}
for moving_secind in range(first_sec+1, last_sec+1):

	# specify transform matrix if elastix failed to align
	if stack == 'MD581' and moving_secind == 32:
		# from 32 to 31, had to do the reverse
		transformation_to_previous_sec[moving_secind] = np.array([[  9.99192973e-01,  -4.01671912e-02,   9.43093242e+01],
       [  4.01671912e-02,   9.99192973e-01,   1.32049088e+00],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

	elif stack == 'MD581' and moving_secind == 12:
		transformation_to_previous_sec[moving_secind] =  np.array([[   1.,   -0.,   156.],
			[   0.,    1.,  144.],
			[   0.,    0.,    1.]])
	else:
	    param_fn = os.path.join(input_dir, 'output%dto%d/TransformParameters.0.txt'%(moving_secind, moving_secind-1))
	    if not os.path.exists(param_fn):
	    	raise Exception('Transform file does not exist: %d to %d' % (moving_secind, moving_secind-1))
	    transformation_to_previous_sec[moving_secind] = parse_parameter_file(param_fn)

# with open(consecutive_transf_filename, 'w') as f:
# 	pickle.dump(transformation_to_previous_sec, f)s

transformation_to_anchor_sec = {}

for moving_secind in range(first_sec, last_sec+1):

	if moving_secind < anchor_sec:
		T_composed = np.eye(3)

		for i in range(anchor_sec, moving_secind, -1):
			T_composed = np.dot(np.linalg.inv(transformation_to_previous_sec[i]), T_composed)
		transformation_to_anchor_sec[moving_secind] = T_composed

	else:
		T_composed = np.eye(3)
		for i in range(anchor_sec+1, moving_secind+1):
			T_composed = np.dot(transformation_to_previous_sec[i], T_composed)
		transformation_to_anchor_sec[moving_secind] = T_composed


with open(final_tranf_filename, 'w') as f:
	pickle.dump(transformation_to_anchor_sec, f)