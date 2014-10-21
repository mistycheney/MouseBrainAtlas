import argparse

parser = argparse.ArgumentParser(description="Run pipeline for different instances on different servers")
parser.add_argument("stack", help="stack name, e.g. RS141")
parser.add_argument("resol", help="resolution, e.g. x5")
parser.add_argument("params", help="parameter set name, e.g. redNissl")
parser.add_argument("start_slice", type=int, help="beginning slice in the stack")
parser.add_argument("end_slice", type=int, help="ending slice in the stack")
args = parser.parse_args()


# def exists_remote(host, path):
#     proc = subprocess.Popen(
#         ['ssh', host, 'test -f %s' % pipes.quote(path)])
#     proc.wait()
#     return proc.returncode == 0


import subprocess
import pipes

def exists_remote(host, path):
    return subprocess.call(['ssh', host, 'test -e ' + pipes.quote(path)]) == 0

hostids = range(31,39) + range(41,49)
n_hosts = len(hostids)

d = {'stack': args.stack, 'resol': args.resol, 'params': args.params}

with open('argfile', 'w') as f:
	for slice_num in range(args.start_slice, args.end_slice + 1):
		d['slice_num'] = slice_num
		if not exists_remote('yuncong@gcn-20-32.sdsc.edu', '/home/yuncong/DavidData/%(stack)s/%(resol)s/%(slice_num)s/%(params)s/pipelineResults'%d):
			f.write('gcn-20-%d.sdsc.edu %04d\n'%(hostids[slice_num%n_hosts], slice_num))

cmd = "parallel --colsep ' ' ssh {1} python /home/yuncong/Brain/notebooks/pipeline_v3.py /home/yuncong/DavidData/%(stack)s/%(resol)s/{2}/%(stack)s_%(resol)s_{2}.tif \
%(params)s -t /home/yuncong/DavidData/RS141/x5/0000/redNissl_pipelineResults/RS141_x5_0000_redNissl_centroids.npy :::: argfile" % d

subprocess.call(cmd, shell=True)
