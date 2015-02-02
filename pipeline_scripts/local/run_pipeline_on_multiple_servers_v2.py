import argparse

parser = argparse.ArgumentParser(description="Run pipeline for different instances on different servers")
parser.add_argument("stack", help="stack name, e.g. RS141")
parser.add_argument("start_slice", type=int, help="beginning slice in the stack")
parser.add_argument("end_slice", type=int, help="ending slice in the stack")
parser.add_argument("-g", "--gabor_params", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
parser.add_argument("-v", "--vq_params", type=str, help="vq parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()

import subprocess
import pipes

# def exists_remote(host, path):
    # return subprocess.call(['ssh', host, 'test -e ' + pipes.quote(path)]) == 0

hostids = range(31,39) + range(41,49)
n_hosts = len(hostids)

# d = {'stack': args.stack, 'resol': args.resol, 'gabor_params': args.gabor_params, 'segm_params': args.segm_params, 'vq_params': args.vq_params, }
d = {'stack': args.stack, 'resol': 'x5', 'gabor_params': args.gabor_params, 'segm_params': args.segm_params, 'vq_params': args.vq_params, 
	'gordon_result_dir': os.environ['GORDON_RESULT_DIR'], 'gordon_data_dir': os.environ['GORDON_DATA_DIR'], 'gordon_repo_dir': os,environ['GORDON_REPO_DIR']}

with open('argfile', 'w') as f:
	for slice_num in range(args.start_slice, args.end_slice + 1):
		d['slice_num'] = slice_num
		f.write('gcn-20-%d.sdsc.edu %d\n'%(hostids[slice_num%n_hosts], slice_num))

cmd = "parallel --colsep ' ' ssh {1} 'python %(gordon_repo_dir)s/pipeline_scripts/pipeline.py %(stack)s {2} -g %(gabor_params)s -s %(segm_params)s -v %(vq_params)s' :::: argfile" % d
print cmd
# subprocess.call(cmd, shell=True)