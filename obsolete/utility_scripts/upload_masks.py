import os
import sys
import subprocess

import argparse

parser = argparse.ArgumentParser(description="Download pipelineResults of a stack of instances")
parser.add_argument("stack", help="stack name, e.g. RS141")
parser.add_argument("resol", help="resolution, e.g. x5")
args = parser.parse_args()


def upload_one_mask(slice_num):

	d = {'obj': None, 'stack':args.stack, 'resol':args.resol, 'slice':slice_num}

	mask_fn = '/home/yuncong/BrainLocal/%(stack)s_%(resol)s_foreground/%(stack)s_%(resol)s_%(slice)s_mask.png'%d

	remote_dir = '/home/yuncong/DavidData/%(stack)s/%(resol)s/%(slice)s/'%d

	cmd = 'scp %s yuncong@gcn-20-32.sdsc.edu:%s`'%(mask_fn, remote_dir)
	print cmd
	subprocess.call(cmd, shell=True)

for slice_num in range(24):
	upload_one_mask('%04d'%slice_num)


