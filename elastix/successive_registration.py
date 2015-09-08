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


if __name__ == '__main__':

	t = time.time()


	stack = sys.argv[1]
	first_sec = int(sys.argv[2])
	last_sec = int(sys.argv[3])


	im_dir = os.path.join('data', stack, stack + '_padded')
	output_dir = create_if_not_exists(os.path.join('data', stack, stack + '_output'))
	transform_params_dir = create_if_not_exists(os.path.join('data', stack, stack + '_transfParams'))
	consecutive_transf_filename = os.path.join('data', stack, stack + '_consecTransfParams.pkl')
	final_tranf_filename = os.path.join('data', stack, stack + '_finalTransfParams.pkl')
	warped_dir = create_if_not_exists(os.path.join('data', stack, stack + '_warped'))

	# im_dir = sys.argv[1]
	# output_dir = create_if_not_exists(sys.argv[2])
	# transform_params_dir = create_if_not_exists(sys.argv[3])
	# consecutive_transf_filename = sys.argv[4]
	# final_tranf_filename = sys.argv[5]
	# warped_dir = create_if_not_exists(sys.argv[6])

	n_sections = len(os.listdir(im_dir))

	os.chdir('/home/yuncong/elastix_linux64_v4.7')

	nr_param = "/parameters/Parameters_BSpline.txt"
	af_param = "parameters/Parameters_Affine.txt"
	rg_param = "parameters/Parameters_Rigid.txt"

	d = {'im_dir': im_dir, 'af_param':af_param, 'nr_param': nr_param, 'rg_param': rg_param, 'output_dir': output_dir, 'transform_params_dir': transform_params_dir}

	create_if_not_exists(d['transform_params_dir'])

	for i in range(first_sec+1, last_sec+1):
		d['f'] = i-1
		d['m'] = i
		create_if_not_exists('%(output_dir)s/output%(m)dto%(f)d'%d)
		execute_command('bin/elastix -f %(im_dir)s/%(f)d.jpg -m %(im_dir)s/%(m)d.jpg -out %(output_dir)s/output%(m)dto%(f)d -p %(rg_param)s' % d)
		# execute_command('bin/elastix -f %(im_dir)s/%(f)d.tif -m %(im_dir)s/%(m)d.tif -out %(output_dir)s/output%(m)dto%(f)d -p %(rg_param)s' % d)
		execute_command('cp %(output_dir)s/output%(m)dto%(f)d/TransformParameters.0.txt %(transform_params_dir)s/TransformParameters_%(m)dto%(f)d.0.txt' % d)

	transformation_to_previous_sec = {}
	for filename in os.listdir(transform_params_dir):

		T = parse_parameter_file(os.path.join(transform_params_dir, filename))

		g = re.match('TransformParameters_(\d+)to(\d+).0.txt', filename.split('/')[-1])
		groups = g.groups()
		moving_secind = int(groups[0])
		fixed_secind = int(groups[1])

		transformation_to_previous_sec[moving_secind] = T

	# with open(consecutive_transf_filename, 'w') as f:
	# 	pickle.dump(transformation_to_previous_sec, f)

	# max_sec = np.max(transformation_to_previous_sec.keys())


	transformation_to_first_sec = {}

	for moving_secind in range(first_sec, last_sec+1):
		T_composed = np.eye(3)
		for i in range(first_sec+1, moving_secind+1):
			T_composed = np.dot(transformation_to_previous_sec[i], T_composed)
		transformation_to_first_sec[moving_secind] = T_composed

	with open(final_tranf_filename, 'w') as f:
		pickle.dump(transformation_to_first_sec, f)


	if not os.path.exists(warped_dir):
		os.makedirs(warped_dir)

	for sec_ind, T in transformation_to_first_sec.iteritems():
		# if sec_ind == 2 or sec_ind == 1 or sec_ind == 0:
		# 	print T
		image = imread(im_dir + '/' + str(sec_ind) + '.jpg')
		# image = imread(im_dir + '/' + str(sec_ind) + '.tif')
		image_warped_to_first_sec = warp(image, inverse_map=AffineTransform(T))
		output_filename = warped_dir + '/' + str(sec_ind) + '.jpg'
		# output_filename = warped_dir + '/' + str(sec_ind) + '.tif'
		imsave(output_filename, image_warped_to_first_sec)


	print '\nAligned %d images in %f seconds' %(n_sections, time.time() - t)