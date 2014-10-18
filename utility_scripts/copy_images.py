import os
import sys
import subprocess

# stack = 'RS141'
stack = sys.argv[1]
# resol = 'x5'
resol = sys.argv[2]

d = {'stack': stack,
'resol' : resol}

for i in range(24):
	d['slice'] = '%04d'%i
	img = '/home/yuncong/BrainLocal/DavidData/%(stack)s/%(resol)s/%(slice)s/%(stack)s_%(resol)s_%(slice)s.tif'%d
	cmd = 'cp %s .' % img
	print cmd
	subprocess.call(cmd, shell=True)