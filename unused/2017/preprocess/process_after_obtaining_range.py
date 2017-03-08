#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Process after identifying the first and last sections in the stack that contain brainstem (if desired, can be whole stack): 1) align thumbnails 2) warp thumbnail images 3) generate mask')

parser.add_argument("stack_name", type=str, help="stack name")
parser.add_argument("-f", "--first_sec", type=int, help="first section")
parser.add_argument("-l", "--last_sec", type=int, help="last section")
parser.add_argument("-p", "--use_precomputed", type=int, help="use precomputed transforms", default=False)
args = parser.parse_args()

import os
import sys

sys.path.append(os.environ['REPO_DIR'] + '/utilities/')
from preprocess_utility import *
from metadata import *

stack = args.stack_name

last_sec = args.last_sec if args.last_sec is not None else section_range_lookup[stack][1]
first_sec = args.first_sec if args.first_sec is not None else section_range_lookup[stack][0]

# first_sec = args.first_sec
# last_sec = section_number_lookup[stack] if args.last_sec is None else args.last_sec
use_precomputed_transforms = args.use_precomputed

from subprocess import check_output
import time
import numpy as np

d = {
     'script_dir': os.path.join(os.environ['REPO_DIR'], 'preprocess'),
     'stack': stack,
     'first_sec': first_sec,
     'last_sec': last_sec,
     'input_dir': os.path.join( os.environ['DATA_DIR'], stack+'_thumbnail_renamed'),
	 'elastix_output_dir': os.path.join( os.environ['DATA_DIR'], stack+'_elastix_output'),
	 'aligned_dir': os.path.join( os.environ['DATA_DIR'], stack+'_thumbnail_aligned'),
	 'suffix': 'thumbnail'
    }


exclude_nodes = [33, 38]

# elastix has built-in parallelism
t = time.time()
print 'aligning...',

if not use_precomputed_transforms:

    # if os.path.exists(d['elastix_output_dir']):
    #     os.system('rm -r ' + d['elastix_output_dir'] + '/*')
    # create_if_not_exists(d['elastix_output_dir'])
    #
    # jump_aligned_sections = {}
    # for moving_secind in range(first_sec+1, last_sec+1):
	# 	for i in range(1, 10):
	# 		if moving_secind - i not in bad_sections[stack]:
	# 			last_good_section = moving_secind - i
	# 			break
	# 	if i != 1:
	# 		jump_aligned_sections[moving_secind] = last_good_section
    #
    # print jump_aligned_sections
    # pickle.dump(jump_aligned_sections, open(os.path.join(d['elastix_output_dir'], 'jump_aligned_sections.pkl'), 'w'))
    #
    # run_distributed3('%(script_dir)s/align_consecutive.py %(stack)s %(input_dir)s %(elastix_output_dir)s %%(f)d %%(l)d'%d,
    #                 first_sec=first_sec,
    #                 last_sec=last_sec,
    #                 stdout=open('/tmp/log', 'ab+'),
    #                 take_one_section=False,
    #                 exclude_nodes=exclude_nodes)
    #
    # print 'done in', time.time() - t, 'seconds'

    ##################################

    from joblib import Parallel, delayed

    def identify_shape(img_fn):
        return map(int, check_output("identify -format %%Wx%%H %s" % os.path.join(d['input_dir'], img_fn), shell=True).split('x'))

    all_files = dict(sorted([(int(img_fn[:-4].split('_')[1]), img_fn) for img_fn in os.listdir(d['input_dir']) if d['suffix'] in img_fn]))

    all_files = dict([(i, all_files[i]) for i in range(first_sec, last_sec+1)])
    shapes = Parallel(n_jobs=16)(delayed(identify_shape)(img_fn) for img_fn in all_files.values())
    img_shapes_map = dict(zip(all_files.keys(), shapes))
    img_shapes_arr = np.array(img_shapes_map.values())
    largest_sec = img_shapes_map.keys()[np.argmax(img_shapes_arr[:,0] * img_shapes_arr[:,1])]
    print 'largest section is ', largest_sec

    # no parallelism
    t = time.time()
    print 'composing transform...',
    os.system("ssh gcn-20-33.sdsc.edu %(script_dir)s/compose_transform_thumbnail.py %(stack)s %(elastix_output_dir)s %(first_sec)d %(last_sec)d"%d + ' ' + str(largest_sec))
    print 'done in', time.time() - t, 'seconds'

else:
    assert os.path.exists(os.path.join(data_dir, stack+'_elastix_output'))

# else: user provide [stack]_finalTransfParams.pkl


# no parallelism
t = time.time()
print 'warping...',

run_distributed3('%(script_dir)s/warp_crop_IM.py %(stack)s %(input_dir)s %(aligned_dir)s %%(f)d %%(l)d %(suffix)s 0 0 2000 1500'%d,
                first_sec=first_sec,
                last_sec=last_sec,
                take_one_section=False,
                stdout=open('/tmp/log', 'ab+'),
                exclude_nodes=exclude_nodes)

print 'done in', time.time() - t, 'seconds'

sys.exit(1)


t = time.time()
sys.stderr.write('generating mask ...')

run_distributed3(command='%(script_path)s %(stack)s %(input_dir)s %%(f)d %%(l)d'%\
                            {'script_path': os.path.join(os.environ['REPO_DIR'], 'preprocess') + '/generate_thumbnail_masks.py',
                            'stack': stack,
                            'input_dir': os.path.join(os.environ['DATA_DIR'], stack+'_thumbnail_aligned')
                            },
                first_sec=first_sec,
                last_sec=last_sec,
                exclude_nodes=exclude_nodes,
                take_one_section=False)

sys.stderr.write('done in %f seconds\n' % (time.time() - t))
