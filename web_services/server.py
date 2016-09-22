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

from enum import Enum

@app.route('/')
def index():
	return "Brainstem WebService"

@app.route('/align')
def align(sorted_filenames, bad_filenames):

	d = {'result': 0}
	return jsonify(**d)

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', use_reloader=False)
