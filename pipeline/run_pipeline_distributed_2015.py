#! /usr/bin/env python

import argparse
from subprocess import check_output, call
import os
import re
import time

from preprocess_utility import *

parser = argparse.ArgumentParser(description="Run pipeline for different instances on different servers")
parser.add_argument("task", type=str, help="task to perform (must be one of filter, segment, rotate_features)")
parser.add_argument("stack", help="stack name, e.g. MD593")
parser.add_argument("-b", type=int, help="beginning slide (default: %(default)s)", default=0)
parser.add_argument("-e", type=int, help="ending slide (default: %(default)s)", default=-1)

parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')

args = parser.parse_args()

t = time.time()

if args.task == 'filter':
	run_distributed3(command='%(script_path)s %(stack)s %%(secind)d -g %(gabor_p)s'%\
								{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/gabor_filter.py', 
								'stack': args.stack,
								'gabor_p': args.gabor_params_id,
								# 'vq_p': args.vq_params_id,
								}, 
					first_sec=args.b,
					last_sec=args.e,
					exclude_nodes=[33])


elif args.task == 'segment':
	run_distributed3(command='%(script_path)s %(stack)s %%(secind)d -s %(segm_p)s'%\
								{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/segment.py', 
								'stack': args.stack,
								'segm_p': args.segm_params_id,
								}, 
					first_sec=args.b,
					last_sec=args.e,
					exclude_nodes=[33])

# elif args.task == 'directionality':
# 	arg_tuples = [(args.stack, i) for i in slide_indices]
# 	run_distributed3(os.environ['GORDON_REPO_DIR']+'/notebooks/compute_structure_tensor_map_executable.py', arg_tuples)

elif args.task == 'compute_textons':
	cmd = "ssh yuncong@gcn-20-34.sdsc.edu '%(script_path)s %(stack)s %(first_sec)d %(last_sec)d %(interval)d -g %(gabor_p)s -v %(vq_p)s'" %\
					{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/compute_textons.py', 
					'first_sec': args.b,
					'last_sec': args.e,
					'stack': args.stack,
					'interval': 5,
					'gabor_p': args.gabor_params_id,
					'vq_p': args.vq_params_id,
					}
	print cmd
	call(cmd, shell=True)

elif args.task == 'compute_histograms':
	run_distributed3(command='%(script_path)s %(stack)s %%(secind)d -g %(gabor_p)s -v %(vq_p)s -s %(segm_p)s'%\
								{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/compute_histogram.py', 
								'stack': args.stack,
								# 'texton_path': os.environ['GORDON_RESULT_DIR'] + '/' + args.stack + '/' + args.stack + '_lossless_textons.npy',
								'gabor_p': args.gabor_params_id,
								'vq_p': args.vq_params_id,
								'segm_p': args.segm_params_id,
								}, 
					first_sec=args.b,
					last_sec=args.e,
					exclude_nodes=[33])

elif args.task == 'grow_regions':
	run_distributed3(command='%(script_path)s %(stack)s %%(secind)d -g %(gabor_p)s -v %(vq_p)s -s %(segm_p)s'%\
								{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/grow_regions.py', 
								'stack': args.stack,
								'gabor_p': args.gabor_params_id,
								'vq_p': args.vq_params_id,
								'segm_p': args.segm_params_id,
								}, 
					first_sec=args.b,
					last_sec=args.e,
					exclude_nodes=[33])

elif args.task == 'find_consensus':
	run_distributed3(command='%(script_path)s %(stack)s %%(secind)d -g %(gabor_p)s -v %(vq_p)s -s %(segm_p)s'%\
								{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/find_consensus.py', 
								'stack': args.stack,
								'gabor_p': args.gabor_params_id,
								'vq_p': args.vq_params_id,
								'segm_p': args.segm_params_id,
								}, 
					first_sec=args.b,
					last_sec=args.e,
					exclude_nodes=[33])

elif args.task == 'match_landmarks':
	run_distributed3(command='%(script_path)s %(stack)s %%(secind)d -g %(gabor_p)s -v %(vq_p)s -s %(segm_p)s'%\
							{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/match_landmarks.py', 
							'stack': args.stack,
							'gabor_p': args.gabor_params_id,
							'vq_p': args.vq_params_id,
							'segm_p': args.segm_params_id,
							}, 
				first_sec=args.b,
				last_sec=args.e,
				exclude_nodes=[33])

elif args.task == 'visualize_atlas':
	run_distributed3(command='%(script_path)s %(stack)s %%(secind)d -g %(gabor_p)s -v %(vq_p)s -s %(segm_p)s'%\
							{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/visualize_atlas.py', 
							'stack': args.stack,
							'gabor_p': args.gabor_params_id,
							'vq_p': args.vq_params_id,
							'segm_p': args.segm_params_id,
							}, 
				first_sec=args.b,
				last_sec=args.e,
				exclude_nodes=[33])

else:
	raise Exception('task must be one of filter, segment, compute_textons, compute_histograms, grow_regions, find_consensus, match_landmarks')


print args.task, time.time() - t, 'seconds'
