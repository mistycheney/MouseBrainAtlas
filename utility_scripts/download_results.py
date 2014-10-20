import os
import sys
import subprocess

import argparse

parser = argparse.ArgumentParser(description="Download pipelineResults of a stack of instances")
parser.add_argument("stack", help="stack name, e.g. RS141")
parser.add_argument("resol", help="resolution, e.g. x5")
parser.add_argument("params", help="parameter set name, e.g. redNissl")
args = parser.parse_args()


objs = ['cropImg.tif', 
'cropMask.npy',
'dirHist.npy',
'segmentation.npy',
'texHist.npy',
'texMap.npy',
'uncropMask.npy',
'fg.npy',
'bg.npy',
'neighbors.npy',
'centroids.npy',
'segmentation.tif',
'texMap.tif']

def download_one_result(slice_num):

	d = {'obj': None, 'stack':args.stack, 'resol':args.resol, 'slice':slice_num, 'params':args.params}

	local_dir = '/home/yuncong/BrainLocal/DavidData/%(stack)s/%(resol)s/%(slice)s/%(params)s/pipelineResults/'%d

	print slice_num, local_dir

	if not os.path.exists(local_dir):
		os.makedirs(local_dir)
	elif len(os.listdir(local_dir)) > 2:
		return

	for obj in objs:
		
		# if any([f.endswith(obj) for f in os.listdir(local_dir)]):
		# 	print obj, 'exists'
		# 	continue

		d['obj'] = obj
		
		remote_file = '/home/yuncong/DavidData/%(stack)s/%(resol)s/%(slice)s/%(params)s/pipelineResults/*%(obj)s'%d

		cmd = 'scp gcn-20-32.sdsc.edu:%s %s'%(remote_file, local_dir)
		print cmd
		subprocess.call(cmd, shell=True)

for slice_num in range(25):
	download_one_result('%04d'%slice_num)


