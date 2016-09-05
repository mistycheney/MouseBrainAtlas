#! /usr/bin/env python

from flask import Flask, jsonify, request

app = Flask(__name__)

import os
import argparse
import sys
import time
import cv2

# parser = argparse.ArgumentParser(
#     formatter_class=argparse.RawDescriptionHelpFormatter,
#     description='Top down detection of specified landmarks')

# parser.add_argument("stack_name", type=str, help="stack name")
# parser.add_argument("slice_ind", type=int, help="slice index")
# parser.add_argument("-l", "--labels", type=str, help="labels", nargs='+', default=[])
# parser.add_argument("-g", "--gabor_params_id", type=str, help="gabor filter parameters id (default: %(default)s)", default='blueNisslWide')
# parser.add_argument("-s", "--segm_params_id", type=str, help="segmentation parameters id (default: %(default)s)", default='blueNisslRegular')
# parser.add_argument("-v", "--vq_params_id", type=str, help="vector quantization parameters id (default: %(default)s)", default='blueNissl')
# args = parser.parse_args()

# print args.labels

sys.path.append(os.path.join(os.environ['GORDON_REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *

@app.route('/')
def index():
	return "Brainstem Web Service"

@app.route('/set_sorted_filenames')
def set_sorted_filenames(sorted_filenames):
	# Generate [stack]_filename_map.txt
	# Run rename.py with infer_order = 0

@app.route('/align')
def align(first_section, last_section, bad_sections):

	elastix_output_dir = ''

    if os.path.exists(elastix_output_dir):
        os.system('rm -r %(elastix_output_dir)s/*' % {'elastix_output_dir': elastix_output_dir})
    create_if_not_exists(elastix_output_dir)

    jump_aligned_sections = {}
    for moving_secind in range(first_section, last_section+1):
		for i in range(1, 10):
			if moving_secind - i not in bad_sections:
				last_good_section = moving_secind - i
				break
		if i != 1:
			jump_aligned_sections[moving_secind] = last_good_section

    print jump_aligned_sections
    pickle.dump(jump_aligned_sections, open(os.path.join(elastix_output_dir, 'jump_aligned_sections.pkl'), 'w'))

    run_distributed3('%(script_dir)s/align_consecutive.py %(stack)s %(input_dir)s %(elastix_output_dir)s %%(f)d %%(l)d'%d,
                    first_sec=first_section,
                    last_sec=last_section,
                    stdout=open('/tmp/log', 'ab+'),
                    take_one_section=False,
                    exclude_nodes=exclude_nodes)

    print 'done in', time.time() - t, 'seconds'

	d = {'result': 0}
	return jsonify(**d)

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', use_reloader=False)
