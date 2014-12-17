import os
import sys
import subprocess

import argparse

parser = argparse.ArgumentParser(description="Download pipelineResults of a stack of instances")
parser.add_argument("stack", help="stack name, e.g. RS141")
parser.add_argument("resol", help="resolution, e.g. x5")
parser.add_argument("start_slice", type=int, help="beginning slice in the stack")
parser.add_argument("end_slice", type=int, help="ending slice in the stack")
args = parser.parse_args()

objs = ['*.jpg', 
'*.pkl',
'*.png',
'*spProps.npy',
'*neighbors.npy',
'*segmentation.npy',
'*texHist.npy']

data_dir = '/home/yuncong/BrainLocal/DavidData_v4/'

def download_one_result(slice_num):

	d = {'obj': None, 'stack':args.stack, 'resol':args.resol, 'slice':slice_num}

	local_dir = data_dir + '/' + '%(stack)s/%(resol)s/%(slice)s/pipelineResults/'%d

	print slice_num, local_dir

	if not os.path.exists(local_dir):
		os.makedirs(local_dir)
	elif len(os.listdir(local_dir)) > 5:
		print len(os.listdir(local_dir))
		return

	for obj in objs:
		
		# if any([f.endswith(obj) for f in os.listdir(local_dir)]):
		# 	print obj, 'exists'
		# 	continue

		d['obj'] = obj
		
		remote_file = '/home/yuncong/DavidData/%(stack)s/%(resol)s/%(slice)s/pipelineResults/%(obj)s'%d

		cmd = 'scp gcn-20-33.sdsc.edu:%s %s'%(remote_file, local_dir)
		print cmd
		subprocess.call(cmd, shell=True)

for slice_num in range(args.start_slice, args.end_slice + 1):
	download_one_result('%04d'%slice_num)


