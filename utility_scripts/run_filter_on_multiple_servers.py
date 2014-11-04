import argparse

parser = argparse.ArgumentParser(description="Run pipeline for different instances on different servers")
parser.add_argument("stack", help="stack name, e.g. RS141")
parser.add_argument("resol", help="resolution, e.g. x5")
parser.add_argument("start_slice", type=int, help="beginning slice in the stack")
parser.add_argument("end_slice", type=int, help="ending slice in the stack")
parser.add_argument("-g", "--gabor_params", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
parser.add_argument("-s", "--segm_params", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNissl')
parser.add_argument("-v", "--vq_params", type=str, help="vq parameters id (default: %(default)s)", default='blueNissl')
args = parser.parse_args()


import subprocess
import pipes

def exists_remote(host, path):
    return subprocess.call(['ssh', host, 'test -e ' + pipes.quote(path)]) == 0

hostids = range(31,39) + range(41,49)
n_hosts = len(hostids)

d = {'stack': args.stack, 'resol': args.resol, 'gabor_params': args.gabor_params, 'segm_params': args.segm_params, }

with open('argfile', 'w') as f:
	for slice_num in range(args.start_slice, args.end_slice + 1):
		d['slice_num'] = slice_num
		result_exists = exists_remote('yuncong@gcn-20-32.sdsc.edu', '/home/yuncong/DavidData/%(stack)s/%(resol)s/%(slice_num)04d/segmResults/%(stack)s_%(resol)s_%(slice_num)04d_gabor-%(gabor_params)s-segm-%(segm_params)s_neighbors.npy'%d)
		if not result_exists:
			# print '/home/yuncong/DavidData/%(stack)s/%(resol)s/%(slice_num)04d/filterResults/%(stack)s_%(resol)s_%(slice_num)04d_gabor-%(gabor_params)s-segm-%(segm_params)s_neighbors.npy'%d
			f.write('gcn-20-%d.sdsc.edu %d\n'%(hostids[slice_num%n_hosts], slice_num))

cmd = "parallel --colsep ' ' ssh {1} 'python /home/yuncong/Brain/notebooks/gabor_filter_release.py %(stack)s %(resol)s {2} -g %(gabor_params)s -s %(segm_params)s' :::: argfile" % d
print cmd
subprocess.call(cmd, shell=True)