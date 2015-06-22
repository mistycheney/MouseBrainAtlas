import argparse

parser = argparse.ArgumentParser(description="Run pipeline for different instances on different servers")
parser.add_argument("stack", help="stack name, e.g. RS141")
parser.add_argument("n_slides", type=int, help="number of slides, use 0 for all slides")
# parser.add_argument("start_section", type=int, help="beginning section in the stack")
# parser.add_argument("end_section", type=int, help="ending section in the stack")
parser.add_argument("-j", "--slides_per_node", type=int, help="number of slides each node processes (default: %(default)d)", default=4)
parser.add_argument("-g", "--gabor_params", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params", type=str, help="vq parameters id (default: %(default)s)", default='blueNissl')

args = parser.parse_args()

import subprocess
import pipes
import os

from preprocess_utility import *

t = time.time()


s = check_output("ssh gordon.sdsc.edu ls %s" % os.path.join(os.environ['GORDON_DATA_DIR'], stack, 'x5'), shell=True)
# print s
slide_indices = [int(re.split("_|-", f[:-5])[1]) for f in s.split('\n') if len(f) > 0]

if n_slides == 0:
	n_slides = max(slide_indices)
	print 'last slide index', n_slides

# hostids = range(31,39) + range(41,49)
# n_hosts = len(hostids)

# d = {'stack': args.stack, 'resol': args.resol, 'gabor_params': args.gabor_params, 'segm_params': args.segm_params, 'vq_params': args.vq_params, }
# d = {'stack': args.stack, 'resol': 'x5', 'gabor_params': args.gabor_params, 'segm_params': args.segm_params, 'vq_params': args.vq_params, 
# 'gordon_result_dir': os.environ['GORDON_RESULT_DIR'], 'gordon_data_dir': os.environ['GORDON_DATA_DIR'], 'gordon_repo_dir': os.environ['GORDON_REPO_DIR']}

script_path = os.path.join(os.environ['GORDON_REPO_DIR'], 'pipeline_scripts', 'distributed', 'gabor_filter.py')
# arg_tuples = [(stack, i, min(i + slides_per_node - 1, n_slides)) 
# 				for i in range(1, n_slides + 1, slides_per_node)]
arg_tuples = [(stack, i) for i in range(1, n_slides + 1, slides_per_node)]
run_distributed3(script_path, arg_tuples)

# run_distributed('segmentation.py')
# run_distributed('rotate_features.py')

# d['section_interval'] = 5
# cmd = "ssh yuncong@gcn-20-33.sdsc.edu 'python %(gordon_repo_dir)s/pipeline_scripts/generate_textons.py %(stack)s %(section_interval)s -g %(gabor_params)s -s %(segm_params)s -v %(vq_params)s'" %d
# print cmd
# subprocess.call(cmd, shell=True)

# run_distributed('assign_textons.py')
# run_distributed('compute_texton_histograms.py')
# run_distributed('grow_regions.py')
# run_distributed('grow_regions_greedy_executable.py')

# run_distributed('match_boundaries_edge_executable.py')

print time.time() - t, 'seconds'