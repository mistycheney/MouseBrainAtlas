import os
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
'segmentation.tif']

for obj in objs:
	cmd = 'scp gcn-20-32.sdsc.edu:/home/yuncong/DavidData/RS141/x5/0001/redNissl/pipelineResults/*%s /home/yuncong/BrainLocal/DavidData/RS141/x5/0001/redNissl/pipelineResults/' %obj
	# print cmd
	subprocess.call(cmd, shell=True)
