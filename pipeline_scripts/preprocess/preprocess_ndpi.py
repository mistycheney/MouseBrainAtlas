#!/usr/bin/python

def execute_command(cmd):
	try:
	    retcode = call(cmd, shell=True)
	    if retcode < 0:
	        print >>sys.stderr, "Child was terminated by signal", -retcode
	    else:
	        print >>sys.stderr, "Child returned", retcode
	except OSError as e:
	    print >>sys.stderr, "Execution failed:", e
	    raise e

import os
import sys
import shutil
import glob
import argparse

sys.path.append('/home/yuncong/morphsnakes')
import morphsnakes

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.io import imread, imsave

ndpi_dir = os.environ['GORDON_NDPI_DIR']
temp_dir = os.environ['GORDON_TEMP_DIR']
data_dir = os.environ['GORDON_DATA_DIR']
repo_dir = os.environ['GORDON_REPO_DIR']
result_dir = os.environ['GORDON_RESULT_DIR']
labeling_dir = os.environ['GORDON_LABELING_DIR']

ndpisplit = '/oasis/projects/nsf/csd181/yuncong/ndpisplit'

def foreground_mask_morphsnakes(img):

	gI = morphsnakes.gborders(img, alpha=20000, sigma=1)

	mgac = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.3, balloon=-3)

	mgac.levelset = np.ones_like(img)
	mgac.levelset[:10,:] = 0
	mgac.levelset[-10:,:] = 0
	mgac.levelset[:,:10] = 0
	mgac.levelset[:, -10:] = 0

	num_iters = 400
	for i in xrange(num_iters):
	    msnake.step()

	return msnake.levelset

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("stack", type=str, help="choose what stack of images to crop and resolution, ex: RS141")
	parser.add_argument("-r","--rotation",type=str,help="how each slice will be rotated",default = None)
	parser.add_argument("-m","--mirror",type=str,help="to mirror horizontal type 'flop', for vertical type 'flip'",default=None)

	args = parser.parse_args()

	stack = args.stack
	stack_temp_dir = os.path.join(temp_dir, stack)
	if not os.path.exists(stack_temp_dir):
		os.makedirs(stack_temp_dir)

	# Split ndpi files

	for ndpi_file in os.listdir(os.path.join(ndpi_dir, stack)):
		if not ndpi_file.endswith('ndpi'): continue
		execute_command(ndpisplit + ' ' + ndpi_file)

		for level in ['macro', 'x0.078125', 'x0.3125', 'x1.25', 'x5', 'x20']:
			res_temp_dir = os.path.join(stack_temp_dir, level)
			os.mkdir(res_temp_dir)
			for f in glob.glob('*_%s_z0.tif'%level):
				shutil.move(f, res_temp_dir)

		map_dir = os.mkdir(os.path.join(stack_temp_dir, 'map'))
		for f in glob.glob('*map*'%level):
			shutil.move(f, res_temp_dir)

	# Crop sections out of whole-slide images, according to manually produced bounding boxes information

    stack_data_dir = os.path.join(data_dir, stack)
    if not os.path.exists(stack_data_dir):
    	os.makedirs(stack_data_dir)

	for resol in ['x0.3125', 'x1.25', 'x5', 'x20']:

	    res_data_dir = os.path.join(stack_data_dir, resol)
	    if not os.path.exists(res_data_dir):
			os.makedirs(res_data_dir)

		section_ind = 0

		res_temp_dir = os.path.join(stack_temp_dir, resol)

		for slide_im_filename in os.listdir(res_temp_dir):
			_, slide_str, _ = slide_im_filename.split('_')[:3]

			slide_im_path = os.path.join(res_temp_dir, slide_im_filename)

		    img_id = subprocess.check_output(["identify", slide_im_path]).split()
		    tot_w, tot_h = map(int, img_id[2].split('x'))

			bb_txt = os.path.join(repo_dir, 'preprocessing/bounding_box_data', stack, stack + '_' + slide_str + '.txt')
			for x_perc, y_perc, w_perc, h_perc in map(split, open(bb_txt, 'r').readlines()):

				(x,y,w,h) = (int(tot_w * float(x_perc)), int(tot_h * float(y_perc)),
							int(tot_w * float(w_perc)), int(tot_h * float(h_perc)))
				
				geom = str(w) + 'x' + str(h) + '+' + str(x) + '+' + str(y)

				section_ind += 1

				section_im_filename = '_'.join([stack, resol, '%04d' % section_ind]) +'.tif'
				section_im_path = os.path.join(res_data_dir, section_im_filename)

		        # Crops the image according to bounding box data
		        cmd1 = "convert %s -crop %s %s" % (slide_im_path, geom, section_im_path)           
		        execute_command(cmd1)
		        
		        # Rotates the cropped image if specified
		        if args.rotation is not None:
		            cmd2 = "convert %s -page +0+0 -rotate %s %s" % (section_im_path, args.rotation, section_im_path)
		            execute_command(cmd2)
		            
		        # Reflects the rotated image if specified
		        if args.mirror is not None:
		            cmd3 = "convert %s -%s %s" % (section_im_path, args.mirror, section_im_path)
		            execute_command(cmd3)

		        print "Processed %s" % section_im_filename

		        section_im = imread(section_im_path)
		        mask = foreground_mask_morphsnakes(section_im)
				
				mask_filename = '_'.join([stack, resol, '%04d' % section_ind]) +'_mask.png'
				mask_path = os.path.join(res_data_dir, mask_filename)
				
				imsave(mask_path, mask)