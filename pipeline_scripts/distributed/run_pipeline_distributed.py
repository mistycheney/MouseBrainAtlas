import argparse

parser = argparse.ArgumentParser(description="Run pipeline for different instances on different servers")
parser.add_argument("stack", help="stack name, e.g. RS141")
parser.add_argument("start_section", type=int, help="beginning section in the stack")
parser.add_argument("end_section", type=int, help="ending section in the stack")
parser.add_argument("-g", "--gabor_params", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params", type=str, help="vq parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

import subprocess
import pipes
import os

# def exists_remote(host, path):
    # return subprocess.call(['ssh', host, 'test -e ' + pipes.quote(path)]) == 0

hostids = range(31,39) + range(41,49)
n_hosts = len(hostids)

# d = {'stack': args.stack, 'resol': args.resol, 'gabor_params': args.gabor_params, 'segm_params': args.segm_params, 'vq_params': args.vq_params, }
d = {'stack': args.stack, 'resol': 'x5', 'gabor_params': args.gabor_params, 'segm_params': args.segm_params, 'vq_params': args.vq_params, 
'gordon_result_dir': os.environ['GORDON_RESULT_DIR'], 'gordon_data_dir': os.environ['GORDON_DATA_DIR'], 'gordon_repo_dir': os.environ['GORDON_REPO_DIR']}

with open('/tmp/argfile', 'w') as f:
	for section_ind in range(args.start_section, args.end_section + 1):
		d['section_ind'] = section_ind
		f.write('gcn-20-%d.sdsc.edu %d\n'%(hostids[section_ind%n_hosts], section_ind))

def run_distributed(script_name):
	cmd = "parallel --colsep ' ' ssh yuncong@{1} 'python %s/pipeline_scripts/" % d['gordon_repo_dir'] + script_name + " %(stack)s {2} -g %(gabor_params)s -s %(segm_params)s -v %(vq_params)s' :::: /tmp/argfile" % d
	print cmd
	subprocess.call(cmd, shell=True)	

# run_distributed('gabor_filter.py')
# run_distributed('segmentation.py')
# run_distributed('rotate_features.py')

# d['section_interval'] = 5
# cmd = "ssh yuncong@gcn-20-33.sdsc.edu 'python %(gordon_repo_dir)s/pipeline_scripts/generate_textons.py %(stack)s %(section_interval)s -g %(gabor_params)s -s %(segm_params)s -v %(vq_params)s'" %d
# print cmd
# subprocess.call(cmd, shell=True)

run_distributed('assign_textons.py')
# run_distributed('compute_texton_histograms.py')
# run_distributed('grow_regions.py')