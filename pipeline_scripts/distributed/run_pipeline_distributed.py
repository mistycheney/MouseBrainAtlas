import argparse
from subprocess import check_output, call
import os
import re
import time

from preprocess_utility import *

parser = argparse.ArgumentParser(description="Run pipeline for different instances on different servers")
parser.add_argument("task", type=str, help="task to perform (must be one of filter, segment, rotate_features)")
parser.add_argument("stack", help="stack name, e.g. RS141")
# parser.add_argument("-n", type=int, help="number of slides, use 0 for all slides (default: %(default)s)", default=0)
parser.add_argument("-b", type=int, help="beginning slide (default: %(default)s)", default=0)
parser.add_argument("-e", type=int, help="ending slide (default: %(default)s)", default=-1)
# parser.add_argument("start_section", type=int, help="beginning section in the stack")
# parser.add_argument("end_section", type=int, help="ending section in the stack")
# parser.add_argument("-j", "--slides_per_node", type=int, help="number of slides each node processes (default: %(default)d)", default=4)
parser.add_argument("-g", "--gabor_params", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params", type=str, help="vq parameters id (default: %(default)s)", default='blueNissl')

args = parser.parse_args()

# n_slides = args.n

# slides_per_node = args.slides_per_node

t = time.time()

s = check_output("ssh gordon.sdsc.edu ls %s" % os.path.join(os.environ['GORDON_DATA_DIR'], args.stack, 'x5'), shell=True)
slide_indices = [int(f) for f in s.split('\n') if len(f) > 0]

end = len(slide_indices) - 1 if args.e == -1 else args.e
slide_indices = [i for i in slide_indices if i >= args.b and i <= end]

script_root = os.environ['GORDON_REPO_DIR']+'/pipeline_scripts/distributed'

if args.task == 'filter':
	arg_tuples = [(args.stack, i) for i in slide_indices]
	run_distributed3(script_root+'/gabor_filter.py', arg_tuples)

elif args.task == 'segment':
	arg_tuples = [(args.stack, i) for i in slide_indices]
	run_distributed3(script_root+'/segmentation.py', arg_tuples)

# elif args.task == 'rotate_features':
# 	arg_tuples = [(args.stack, i) for i in range(n_slides)]
# 	run_distributed3(script_root+'/rotate_features.py', arg_tuples)

elif args.task == 'generate_textons':
	section_interval = 5
	cmd = "ssh yuncong@gcn-20-33.sdsc.edu 'python %s/generate_textons.py %s %d'" %(script_root, args.stack, section_interval)
	print cmd
	call(cmd, shell=True)

elif args.task == 'assign_textons':
	arg_tuples = [(args.stack, i, os.environ['GORDON_RESULT_DIR']+'/RS141/RS141_x5_gabor-blueNisslWide-vq-blueNissl_textons.npy') for i in slide_indices]
	run_distributed3(script_root+'/assign_textons.py', arg_tuples)

elif args.task == 'compute_texton_histograms':
	arg_tuples = [(args.stack, i) for i in slide_indices]
	run_distributed3(script_root+'/compute_texton_histograms.py', arg_tuples)

elif args.task == 'grow_regions':
	arg_tuples = [(args.stack, i) for i in slide_indices]
	run_distributed3(script_root+'/grow_regions_greedy_executable.py', arg_tuples)

elif args.task == 'detect_edges':
	arg_tuples = [(args.stack, i) for i in slide_indices]
	run_distributed3(script_root+'/detect_edges_executable.py', arg_tuples)

elif args.task == 'match_landmarks':
	arg_tuples = [(args.stack, i) for i in slide_indices]
	run_distributed3(script_root+'/match_landmarks_executable.py', arg_tuples)

elif args.task == 'compute_edgemap':
	arg_tuples = [(args.stack, i) for i in slide_indices]
	run_distributed3(script_root+'/compute_edgemap.py', arg_tuples)

else:
	raise Exception('task must be one of filter, segment, generate_textons, assign_textons, compute_texton_histograms, grow_regions, detect_edges, match_landmarks')


print args.task, time.time() - t, 'seconds'
