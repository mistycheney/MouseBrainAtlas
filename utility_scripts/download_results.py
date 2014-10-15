import os
import sys
import subprocess

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

stack, resol, slice, params = sys.argv[1:]
d = {'obj': None, 'stack':stack, 'resol':resol, 'slice':slice, 'params':params}

local_dir = '/home/yuncong/BrainLocal/DavidData/%(stack)s/%(resol)s/%(slice)s/%(params)s/pipelineResults/'%d

if not os.path.exists(local_dir):
	os.makedirs(local_dir)

for obj in objs:
	
	# if any([f.endswith(obj) for f in os.listdir(local_dir)]):
	# 	print obj, 'exists'
	# 	continue

	d['obj'] = obj
	
	remote_file = '/home/yuncong/DavidData/%(stack)s/%(resol)s/%(slice)s/%(params)s/pipelineResults/*%(obj)s'%d

	cmd = 'scp gcn-20-32.sdsc.edu:%s %s'%(remote_file, local_dir)
	print cmd
	subprocess.call(cmd, shell=True)
