#!/usr/bin/env python

from subprocess import check_output, call
import os 
import numpy as np
import sys
from skimage.io import imread, imsave

from joblib import Parallel, delayed

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

if __name__ == '__main__':

	im_dir = sys.argv[1]
	out_dir = sys.argv[2]
	margin = int(sys.argv[3])

	filenames = os.listdir(im_dir)

	ext = filenames[0][-3:]
	print ext

	all_files = sorted([img_fn for img_fn in filenames if img_fn.endswith(ext)], key=lambda x: int(x.split('.')[0]))

	img_shapes = np.array([map(int, check_output("identify %s" % os.path.join(im_dir, img_fn), shell=True).split()[2].split('x')) for img_fn in all_files])

	max_width = img_shapes[:,0].max()
	max_height = img_shapes[:,1].max()

	canvas_width = max_width + 2 * margin
	canvas_height = max_height + 2 * margin

	canvas_imgs_path = create_if_not_exists(out_dir)
	# canvas_mask_path = create_if_not_exists(outmask_dir)

	def pad_image(i, img_fn):

		img = imread(os.path.join(im_dir, img_fn))

		if img.ndim == 3:
			bg_color = np.uint8([np.argmax(np.bincount(img[:,:,0].flatten())), np.argmax(np.bincount(img[:,:,1].flatten())), np.argmax(np.bincount(img[:,:,2].flatten()))])
			bg = bg_color * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)
			img[(img[...,0]==255) & (img[...,1]==255) & (img[...,0]==255)] = bg_color
		else:
			bg_color = np.uint8(np.argmax(np.bincount(img.flatten())))
			bg = bg_color * np.ones((canvas_height, canvas_width), dtype=np.uint8)
			img[(img==255)] = bg_color

		# bg_color = np.mean(np.vstack([img[:10,:].reshape((-1,3)), img[-10:,:].reshape((-1,3)), 
		# 	img[:,:10].reshape((-1,3)), img[:,-10:].reshape((-1,3))]), axis=0).astype(np.uint8)

		# alpha = img[..., -1] > 0
		# rgb = img[..., :3]

		# rgb[~alpha] = 255

		img_h, img_w = img.shape[:2]
		bg[canvas_height/2-img_h/2:canvas_height/2+(img_h-img_h/2),
			canvas_width/2-img_w/2:canvas_width/2+(img_w-img_w/2)] = img

		# canvas_mask = np.zeros((canvas_height, canvas_width), dtype=np.bool)
		# canvas_mask[canvas_height/2-img_h/2:canvas_height/2+(img_h-img_h/2),
		# 	canvas_width/2-img_w/2:canvas_width/2+(img_w-img_w/2)] = alpha

		imsave(canvas_imgs_path + '/' + str(i) + '.' + ext, bg)
		# imsave(canvas_mask_path + '/' + str(i) + '.png', canvas_mask.astype(np.uint8)*255)

	execute_command('mogrify -format jpg ' + canvas_imgs_path + '/*.' + ext)
	execute_command('rm ' + canvas_imgs_path + '/*.' + ext)


	Parallel(n_jobs=16)(delayed(pad_image)(i, img_fn) for i, img_fn in enumerate(all_files))

	

