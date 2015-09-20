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

args = parser.parse_args()

t = time.time()

# s = check_output("ssh gordon.sdsc.edu ls %s" % os.path.join(os.environ['GORDON_DATA_DIR'], args.stack, 'lossless'), shell=True)
# slide_indices = [int(f) for f in s.split('\n') if len(f) > 0]

# end = len(slide_indices) - 1 if args.e == -1 else args.e
# slide_indices = [i for i in slide_indices if i >= args.b and i <= end]

# slide_indices = range(args.b, args.e+1)

if args.task == 'filter':
	run_distributed3(command='%(script_path)s %(stack)s %%(secind)d'%\
								{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/gabor_filter.py', 
								'stack': args.stack
								}, 
					first_sec=args.b,
					last_sec=args.e,
					exclude_nodes=[33])


elif args.task == 'segment':
	run_distributed3(command='%(script_path)s %(stack)s %%(secind)d'%\
								{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/segment.py', 
								'stack': args.stack
								}, 
					first_sec=args.b,
					last_sec=args.e,
					exclude_nodes=[33])

# elif args.task == 'directionality':
# 	arg_tuples = [(args.stack, i) for i in slide_indices]
# 	run_distributed3(os.environ['GORDON_REPO_DIR']+'/notebooks/compute_structure_tensor_map_executable.py', arg_tuples)

elif args.task == 'compute_textons':
	cmd = "ssh yuncong@gcn-20-34.sdsc.edu '%(script_path)s %(stack)s %(first_sec)d %(last_sec)d %(interval)d'" %\
					{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/compute_textons.py', 
					'first_sec': args.b,
					'last_sec': args.e,
					'stack': args.stack,
					'interval': 5}

	print cmd
	call(cmd, shell=True)

elif args.task == 'compute_histograms':
	run_distributed3(command='%(script_path)s %(stack)s %%(secind)d %(texton_path)s'%\
								{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/compute_histogram.py', 
								'stack': args.stack,
								'texton_path': os.environ['GORDON_RESULT_DIR'] + '/' + args.stack + '/' + args.stack + '_lossless_textons.npy'
								}, 
					first_sec=args.b,
					last_sec=args.e,
					exclude_nodes=[33])

elif args.task == 'grow_regions':
	run_distributed3(command='%(script_path)s %(stack)s %%(secind)d'%\
								{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/grow_regions.py', 
								'stack': args.stack
								}, 
					first_sec=args.b,
					last_sec=args.e,
					exclude_nodes=[33])

elif args.task == 'find_consensus_edges':
	run_distributed3(command='%(script_path)s %(stack)s %%(secind)d'%\
								{'script_path': os.environ['GORDON_PIPELINE_SCRIPT_DIR']+'/find_consensus_edges.py', 
								'stack': args.stack
								}, 
					first_sec=args.b,
					last_sec=args.e,
					exclude_nodes=[33])

# elif args.task == 'match_landmarks':
# 	arg_tuples = [(args.stack, i) for i in slide_indices]
# 	run_distributed3(script_root+'/match_landmarks_executable.py', arg_tuples)

# elif args.task == 'compute_edgemap':
# 	arg_tuples = [(args.stack, i) for i in slide_indices]
# 	run_distributed3(script_root+'/compute_edgemap.py', arg_tuples)

else:
	raise Exception('task must be one of filter, segment, generate_textons, assign_textons, compute_texton_histograms, grow_regions, detect_edges, match_landmarks')


print args.task, time.time() - t, 'seconds'
