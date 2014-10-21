import os
import sys
import subprocess

import argparse

parser = argparse.ArgumentParser(description="Download pipelineResults of a stack of instances")
parser.add_argument("stack", help="stack name, e.g. RS141")
parser.add_argument("resol", help="resolution, e.g. x5")
parser.add_argument("params", help="parameter set name, e.g. redNissl")
parser.add_argument("start_slice", type=int, help="beginning slice in the stack")
parser.add_argument("end_slice", type=int, help="ending slice in the stack")

args = parser.parse_args()


def remove_one_pipelineResults(slice_num):

	d = {'obj': None, 'stack':args.stack, 'resol':args.resol, 'slice':slice_num, 'params':args.params}

	res_dir = '/home/yuncong/DavidData/%(stack)s/%(resol)s/%(slice)s/%(params)s_pipelineResults'%d

	cmd = 'ssh yuncong@gcn-20-32.sdsc.edu rm -r %s'%(res_dir)
	print cmd
	subprocess.call(cmd, shell=True)

for slice_num in range(args.start_slice, args.end_slice + 1):
	remove_one_pipelineResults('%04d'%slice_num)


