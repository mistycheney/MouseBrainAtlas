import argparse
from subprocess import check_output
import os
import re
import time

from preprocess_utility import *

parser = argparse.ArgumentParser(description="Run pipeline for different instances on different servers")
parser.add_argument("task", type=str, help="task to perform (must be one of filter, segment, rotate_features)")
parser.add_argument("stack", help="stack name, e.g. RS141")
parser.add_argument("n_slides", type=int, help="number of slides, use 0 for all slides")
# parser.add_argument("start_section", type=int, help="beginning section in the stack")
# parser.add_argument("end_section", type=int, help="ending section in the stack")
parser.add_argument("-j", "--slides_per_node", type=int, help="number of slides each node processes (default: %(default)d)", default=4)
parser.add_argument("-g", "--gabor_params", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params", type=str, help="vq parameters id (default: %(default)s)", default='blueNissl')

args = parser.parse_args()

n_slides = args.n_slides
slides_per_node = args.slides_per_node

t = time.time()

s = check_output("ssh gordon.sdsc.edu ls %s" % os.path.join(os.environ['GORDON_DATA_DIR'], args.stack, 'x5'), shell=True)
# print s
slide_indices = [int(f) for f in s.split('\n') if len(f) > 0]

if n_slides == 0:
	n_slides = len(slide_indices)
	print 'number of slides', n_slides

if args.task == 'filter':
	script_path = os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts', 'distributed', 'gabor_filter.py')
	arg_tuples = [(args.stack, i) for i in range(n_slides)]
	run_distributed3(script_path, arg_tuples)

elif args.task == 'segment':
	script_path = os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts', 'distributed', 'segmentation.py')
	arg_tuples = [(args.stack, i) for i in range(n_slides)]
	run_distributed3(script_path, arg_tuples)

elif args.task == 'rotate_features':
	script_path = os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts', 'distributed', 'rotate_features.py')
	arg_tuples = [(args.stack, i) for i in range(n_slides)]
	run_distributed3(script_path, arg_tuples)

# d['section_interval'] = 5
# cmd = "ssh yuncong@gcn-20-33.sdsc.edu 'python %(gordon_repo_dir)s/pipeline_scripts/generate_textons.py %(stack)s %(section_interval)s -g %(gabor_params)s -s %(segm_params)s -v %(vq_params)s'" %d
# print cmd
# subprocess.call(cmd, shell=True)

# run_distributed('assign_textons.py')
# run_distributed('compute_texton_histograms.py')
# run_distributed('grow_regions.py')
# run_distributed('grow_regions_greedy_executable.py')

# run_distributed('match_boundaries_edge_executable.py')

else:
	raise Exception('task must be one of filter, segment, rotate_features')


print args.task, time.time() - t, 'seconds'