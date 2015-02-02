import os
import sys
import subprocess

"""
Copy relevant pipeline result images to the current directory
"""

# stack = 'RS141'
stack = sys.argv[1]
# resol = 'x5'
resol = sys.argv[2]
params = sys.argv[3]

d = {'stack': stack,
'resol' : resol,
'params' : params}

for i in range(24):
	d['slice'] = '%04d'%i
	img = '/home/yuncong/BrainLocal/DavidData_v3/%(stack)s/%(resol)s/%(slice)s/%(params)s_pipelineResults/%(stack)s_%(resol)s_%(slice)s_%(params)s_texMap.tif'%d
	cmd = 'cp %s .' % img
	print cmd
	subprocess.call(cmd, shell=True)