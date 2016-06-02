import argparse
from joblib import Parallel, delayed
import numpy as np

from preprocess_utility import *
from preprocess_movedata import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("stack", type=str, help="choose what stack of images to crop and resolution, ex: RS141")
	parser.add_argument("start_slide", type=int, help="first slide")
	parser.add_argument("end_slide", type=int, help="last slide")
	args = parser.parse_args()

	stack = args.stack

	sections_dir_stack = create_if_not_exists(os.path.join(os.environ['GORDON_SECTIONDATA_DIR'], stack))	# DavidData2015sections/CC35

	autogen_bbox_dir = os.path.join(os.environ['GORDON_SECTIONDATA_DIR'], stack, 'autogen_bbox_x0.3125')

	import tempfile
	import os
	from joblib import load, dump

	arg_tuples = []

	# temp_folder = tempfile.mkdtemp()
	# print temp_folder

	# import time
	# b = time.time()

	# for resol in ['x0.3125', 'x1.25', 'x5', 'x20']:
	for resol in ['x0.3125', 'x1.25', 'x5']:
	# for resol in ['x5']:
	# for resol in ['x0.3125', 'x1.25']:

		create_if_not_exists(os.path.join(sections_dir_stack, resol))

		for slide_ind in range(args.start_slide, args.end_slide + 1):
		
			curr_resol_slide_path = os.path.join(os.environ['GORDON_SLIDEDATA_DIR'], stack, resol, "%s_%02d_%s_z0.tif" % (stack, slide_ind, resol))
			slide_img = np.asarray(Image.open(curr_resol_slide_path).convert('RGB'))

			# if resol == 'x5':
			# 	filename = os.path.join(temp_folder, "%s_%02d_%s_z0" % (stack, slide_ind, resol) + '.mmap')
		 # 		if os.path.exists(filename): os.unlink(filename)
			# 	_ = dump(slide_img, filename)
			# 	slide_img_memmap = load(filename, mmap_mode='r+')

			# 	del slide_img
			# 	import gc
			# 	_ = gc.collect()

			bbox_file = '_'.join(['bbox', stack, 'x0.3125', '%02d'%slide_ind]) + '.txt'
			bboxes_x03125 = np.loadtxt(os.path.join(autogen_bbox_dir, bbox_file)).astype(np.int)
			
			for sec_ind, box_x03125 in enumerate(bboxes_x03125):

				if (box_x03125 < 0).all(): continue

				arg_tuples.append({'stack': stack,
								'slide_ind': slide_ind,
								'resol': resol,
								'bbox_x03125': box_x03125,
								'section_ind': sec_ind,
								'slide_img': slide_img})
								# 'slide_img': slide_img_memmap if resol == 'x5' else slide_img})


	# print time.time() - b

	# b = time.time()	

	# Parallel(n_jobs=16)(delayed(refine_save_highres_maskedimgs_parallel)(**args) 
	# 					for args in arg_tuples)

	Parallel(n_jobs=16, max_nbytes=1e6)(delayed(refine_save_highres_maskedimgs_parallel)(**args)
							for args in arg_tuples)

	# Parallel(n_jobs=16, max_nbytes=None)(delayed(refine_save_highres_maskedimgs_parallel)(**args)
	# 						for args in arg_tuples)

	# for args in arg_tuples:
	# 	refine_save_highres_maskedimgs_parallel(**args)

	# print time.time() - b