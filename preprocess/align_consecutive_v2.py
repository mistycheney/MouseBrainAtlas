#!/usr/bin/env python

import os
import sys
import json

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from metadata import *
from utilities2015 import *

stack = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
filename_pairs = json.loads(sys.argv[4])
fmt = sys.argv[5]
suffix = 'thumbnail'

parameter_dir = os.path.join(os.environ['REPO_DIR'], 'preprocess', 'parameters')

rg_param_rigid = os.path.join(parameter_dir, "Parameters_Rigid.txt")
rg_param_mutualinfo = os.path.join(parameter_dir, "Parameters_Rigid_MutualInfo.txt")

if stack in all_alt_nissl_ntb_stacks or stack in all_alt_nissl_tracing_stacks:
	rg_param = rg_param_mutualinfo
else:
	rg_param = rg_param_rigid

failed_pairs = []

for fn_pair in filename_pairs:
	prev_fn = fn_pair['prev_fn']
	curr_fn = fn_pair['curr_fn']

	output_subdir = os.path.join(output_dir, curr_fn + '_to_' + prev_fn)

	if os.path.exists(output_subdir) and 'TransformParameters.0.txt' in os.listdir(output_subdir):
		sys.stderr.write('Result for aligning %s to %s already exists.\n' % (curr_fn, prev_fn))
		continue

	execute_command('rm -rf \"%s\"' % output_subdir)
	create_if_not_exists(output_subdir)

	ret = execute_command('%(elastix_bin)s -f \"%(fixed_fn)s\" -m \"%(moving_fn)s\" -out \"%(output_subdir)s\" -p \"%(rg_param)s\"' % \
			{'elastix_bin': ELASTIX_BIN,
			'rg_param': rg_param,
			'output_subdir': output_subdir,
			'fixed_fn': os.path.join(input_dir, prev_fn + '.' + fmt),
			'moving_fn': os.path.join(input_dir, curr_fn + '.' + fmt)
			})

	if ret == 1:
		# sys.stderr.write(prev_fn + ' vs. ' + curr_fn + ' failed.\n')
		failed_pairs.append((prev_fn, curr_fn))

import subprocess
hostname = subprocess.check_output("hostname", shell=True).strip()

if len(failed_pairs) > 0:
    with open(os.path.join(output_dir, '%s_failed_pairs_%s.txt' % (stack, hostname.split('.')[0])), 'w') as f:
        for pf, cf in failed_pairs:
            f.write(pf + ' ' + cf + '\n')
