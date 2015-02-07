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

sys.path.append(os.environ['MSNAKES_PATH'])
import morphsnakes

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops, label

from joblib import Parallel, delayed

ndpisplit = os.environ['GORDON_NDPISPLIT_PROGRAM']

def foreground_mask_morphsnakes_slide(img, max_iters=1000, num_section_per_slide=5, min_iter=300):

    gI = morphsnakes.gborders(img, alpha=20000, sigma=1)

    msnake = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.3, balloon=-3)

    msnake.levelset = np.ones_like(img)
    msnake.levelset[:3,:] = 0
    msnake.levelset[-3:,:] = 0
    msnake.levelset[:,:3] = 0
    msnake.levelset[:,-3:] = 0

    for i in xrange(max_iters):
        msnake.step()
        if i > 0:
            diff = np.count_nonzero(msnake.levelset - previous_levelset < 0)
            if diff < 40 and i > min_iter: # oscillate
                break

        previous_levelset = msnake.levelset

    blob_labels, n_labels = label(msnake.levelset, neighbors=4, return_num=True, background=0)

    blob_props = regionprops(blob_labels + 1)
    all_areas = np.array([p.area for p in blob_props])
    all_centers = np.array([p.centroid for p in blob_props])
    all_bboxes = np.array([p.bbox for p in blob_props])

    indices = np.argsort(all_areas)[::-1]
    largest_area = all_areas[indices[:2]].mean()

    valid = np.where(all_areas > largest_area * .45)[0]

    valid = valid[np.argsort(all_centers[valid, 1])]
    centers_x = all_centers[valid, 1]
    centers_y = all_centers[valid, 0]

    height, width = img.shape[:2]


    if len(valid) > num_section_per_slide:
        indices_close = np.where(np.diff(centers_x) < width * .1)[0]

        ups = []
        downs = []

        if len(indices_close) > 0:

            for i in range(len(valid)):
                if i-1 in indices_close:
                    continue
                elif i in indices_close:
                    if centers_y[i] > height * .5:
                        ups.append(valid[i+1])
                        downs.append(valid[i])
                    else:
                        ups.append(valid[i])
                        downs.append(valid[i+1])
                else:
                    if centers_y[i] > height * .5:
                        ups.append(-1)
                        downs.append(valid[i])
                    else:
                        ups.append(valid[i])
                        downs.append(-1)

        arrangement = np.r_[ups, downs]

    elif len(valid) < num_section_per_slide:
        snap_to_columns = (np.round((centers_x / width + 0.1) * num_section_per_slide) - 1).astype(np.int)

        arrangement = -1 * np.ones((num_section_per_slide,), dtype=np.int)
        arrangement[snap_to_columns] = valid
    else:
        arrangement = valid

    bboxes = []
    masks = []

    for i, j in enumerate(arrangement):
        if j == -1: continue

        minr, minc, maxr, maxc = all_bboxes[j]
        bboxes.append(all_bboxes[j])

        mask = np.zeros_like(img, dtype=np.bool)
        mask[blob_labels == j] = 1

        section_mask = mask[minr:maxr+1, minc:maxc+1]

        masks.append(section_mask)

    return masks, bboxes, arrangement > -1

def gen_mask(in_dir=None, slide_ind=None, stack=None, use_hsv=True, out_dir=None, num_section_per_slide=5, mirror=None, rotate=None):

    imgcolor = Image.open(ps.path.join(in_dir, '%s_%02d_x0.3125_z0.tif'%(stack, slide_ind)))
    imgcolor = imgcolor.convert('RGB')
    imgcolor = np.array(imgcolor)
    img_gray = rgb2gray(imgcolor)
    slide_height, slide_width = img_gray.shape[:2]

    if use_hsv:
        imghsv = rgb2hsv(imgcolor)
        img = imghsv[...,1]
    else:
        img = img_gray

    section_masks, bboxes = foreground_mask_morphsnakes_slide(img, num_section_per_slide=num_section_per_slide)
        
#     section_images = []
    for i, ((minr, minc, maxr, maxc), mask) in enumerate(zip(bboxes, section_masks)):
        img = imgcolor[minr:maxr+1, minc:maxc+1].copy()
        img[~mask] = 0
#         section_images.append(img)
        
        image_filepath = os.path.join(out_dir, '%s_x0.3125_%02d_%d.tif'%(stack, slide_ind, i))
        imsave(image_filepath, img)

        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
    return section_masks, bboxes

def create_if_not_exists(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return path

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("stack", type=str, help="choose what stack of images to crop and resolution, ex: RS141")

	args = parser.parse_args()

	stack = args.stack
	args_dict['stack'] = stack

	slide_dir_stack = create_if_not_exists(os.path.join(os.environ['GORDON_TEMP_DIR'], stack))	# DavidData2015temp/CC35

	######################## Split ndpi files ########################

	for ndpi_file in os.listdir(os.path.join(os.environ['GORDON_NDPI_DIR'], stack)):
		if not ndpi_file.endswith('ndpi'): continue
		execute_command(ndpisplit + ' ' + ndpi_file)

		for level in ['macro', 'x0.078125', 'x0.3125', 'x1.25', 'x5', 'x20']:
			slide_dir_resol = os.path.join(slide_dir_stack, level)			# DavidData2015temp/CC35/x0.3125
			os.mkdir(slide_dir_resol)
			for f in glob.glob('*_%s_z0.tif'%level):
				shutil.move(f, slide_dir_resol)

		map_dir = os.mkdir(os.path.join(slide_dir_stack, 'map'))			# DavidData2015temp/CC35/map
		for f in glob.glob('*map*'%level):
			shutil.move(f, map_dir)

    ####################### Section bounding box and mask generation ##########################

   	masked_dir = create_if_not_exists(os.path.join(slide_dir_stack, 'masked_x0.3125'))

	# n_slides = len([f for f in os.listdir(masked_dir) if f.endswith('.tif')])
	n_slides = 3

	options = [{'use_hsv':True, 'in_dir', 'out_dir':masked_dir, 'num_section_per_slide': 5} for _ in range(n_slides)] # set to True for Nissl stains, False for Fluorescent stains

	result = Parallel(n_jobs=16)(delayed(gen_mask)(**kwargs) 
									for (i, kwargs) in zip(range(0, n_slides), options))

	masked_sections, bboxes_percent = *zip(result)

    bbox_dir = create_if_not_exists(os.environ['GORDON_BBOX_DIR'])

	for slide_ind, bboxes_slide in enumerate(bboxes_percent, masked):
		bbox_filepath = os.path.join(bbox_dir, 'bbox_%s_%d.txt'%(stack, slide_ind))
		np.savetxt(bbox_filepath, bboxes_slide)

    ####################### Manually inspect masked sections and bounding boxes; Modify (using Gimp) if necessary ##########################

    ####################### Generate masks and  move cropped images to Data dir ##########################

 #    n_section = 0

 #    os.chdir(bbox_dir)
	# for bbox_file in os.listdir('.'):
	# 	bboxes_slide = np.loadtxt(bbox_file)
	# 	n_section += len(bboxes_slide)




	# data_dir_stack = create_if_not_exists(os.path.join(os.environ['GORDON_SECTIONDATA_DIR'], stack))	# DavidData2015tif/CC35
    
 #    resol = 'x0.3125'
	# args_dict['resol'] = resol

 #    data_dir_resol = create_if_not_exists(os.path.join(data_dir_stack, resol))		# DavidData2015tif/CC35/x0.3125
	# data_dir_images = create_if_not_exists(os.path.join(data_dir_resol, 'images'))	# DavidData2015tif/CC35/x0.3125/images	
	# data_dir_masks = create_if_not_exists(os.path.join(data_dir_resol, 'masks'))		# DavidData2015tif/CC35/x0.3125/masks

	# if resol == 'x0.3125':
	#     os.chdir(masked_dir)
	#     for input_img in os.listdir('.'):
	#     	if not input_img.endswith('.tif'): continue
	#     	mask_path = os.path.join(data_dir_masks, '%(stack)s_%(resol)s_%(section_ind)04d_mask.png' % args_dict)		# DavidData2015tif/CC35/x0.3125/masks/CC35_x0.3125_0001_mask.png
	# 	    execute_command('convert -channel Alpha %s -separate %s' % (input_img, mask_path))
	# 	    shutil.copy(input_img, data_dir_images)

	# else:
	# 	slide_dir_resol = os.path.join(slide_dir_stack, resol)

	# 	section_counter = 0

	# 	for bbox_file in os.listdir(bbox_dir):
	# 		if bbox_file.endswith('txt'):
	# 			slide_ind = int(bbox_file[:-4].split('_')[2])
	# 			bboxes_percent = np.loadtxt(bbox_file)

	# 			slide_imagepath = os.path.join(slide_dir_resol, "CC35_%02d_x0.3125_z0.tif" % slide_ind)		# DavidData2015temp/CC35/x0.3125/CC35_45_x0.3125_z0.tif
	# 			slide_image = np.array(Image.open(slide_imagepath).convert('RGB'))
	# 			slide_width, slide_height = slide_image.shape[:2]

	# 			for minr, minc, maxr, maxc in bboxes_percent:
	# 				minr = int(minr * slide_height)
	# 				maxr = int(maxr * slide_height)
	# 				minc = int(minc * slide_width)
	# 				maxc = int(maxc * slide_width)
	# 				section_image = slide_image[minr:maxr+1, minc:maxc+1]

	# 				args_dict['section_ind'] = section_counter

	# 				imsave(os.path.join(data_dir_images,  "%(stack)s_%(resol)s_%(section_ind)04d.tif" % args_dict))		# DavidData2015tif/CC35/x5/images/CC35_x5_0001.tif

	# 		    	in_mask = os.path.join(data_dir_stack, 'x0.3125', 'masks', '%(stack)s_%(resol)s_%(section_ind)04d_mask.png' % args_dict)	# DavidData2015tif/CC35/x0.3125/masks/CC35_x0.3125_0001_mask.png
	# 		    	out_mask = os.path.join(data_dir_masks, '%(stack)s_%(resol)s_%(section_ind)04d_mask.png' % args_dict)	# DavidData2015tif/CC35/x0.3125/masks/CC35_x0.3125_0001_mask.png
	# 	    		execute_command('convert -resize 400\% %s %s' % (in_mask, out_mask))

	# 				section_counter += 1