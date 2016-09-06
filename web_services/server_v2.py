#! /usr/bin/env python

from flask import Flask, jsonify, request

app = Flask(__name__)

import os
import argparse
import sys
import time
from subprocess import check_output
from joblib import Parallel, delayed

# import cv2

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

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from data_manager import *
from preprocess_utility import *

script_dir = os.path.join(os.environ['REPO_DIR'], 'preprocess')

exclude_nodes = [33]

@app.route('/')
def index():
    return "Brainstem Web Service"

@app.route('/set_sorted_filenames')
def set_sorted_filenames():

    stack = request.args.get('stack', type=str)
    sorted_filenames = request.args.getlist('sorted_filenames')

    with open(data_dir + '/%(stack)s_filename_map.txt' % {'stack':stack}, 'w') as f:
        for i, fn in enumerate(sorted_filenames):
            f.write(fn + ' ' + str(i+1) + '\n')
    execute_command('%(script_dir)s/rename.py %(stack)s 0' % {'script_dir': script_dir, 'stack': stack})

    d = {'result': 0}
    return jsonify(**d)

@app.route('/align')
def align():

    t = time.time()
    print 'aligning...',

    stack = request.args.get('stack', type=str)
    first_section = request.args.get('first_section', type=int)
    last_section = request.args.get('last_section', type=int)
    bad_sections = map(int, request.args.getlist('bad_sections'))
    print 'bad_sections', bad_sections

    elastix_output_dir = os.path.join(data_dir, stack+'_elastix_output')

    if os.path.exists(elastix_output_dir):
        os.system('rm -r %(elastix_output_dir)s/*' % {'elastix_output_dir': elastix_output_dir})
    create_if_not_exists(elastix_output_dir)

    jump_aligned_sections = {}
    for moving_secind in range(first_section, last_section+1):
        i = 1
        while True:
            if moving_secind - i not in bad_sections:
                last_good_section = moving_secind - i
                break
            i += 1
        if i != 1:
            jump_aligned_sections[moving_secind] = last_good_section

    print jump_aligned_sections
    pickle.dump(jump_aligned_sections, open(os.path.join(elastix_output_dir, 'jump_aligned_sections.pkl'), 'w'))

    input_dir = os.path.join(data_dir, stack+'_thumbnail_renamed')

    run_distributed3('%(script_dir)s/align_consecutive.py %(stack)s %(input_dir)s %(elastix_output_dir)s %%(f)d %%(l)d %(bad_secs)s' % \
                    {'stack': stack,
                    'script_dir': script_dir,
                    'input_dir': input_dir,
                    'elastix_output_dir': elastix_output_dir,
                    'bad_secs': '_'.join(map(str, bad_sections))},
                    first_sec=first_section,
                    last_sec=last_section,
                    stdout=open('/tmp/log', 'ab+'),
                    take_one_section=False,
                    exclude_nodes=exclude_nodes)

    print 'done in', time.time() - t, 'seconds'

    ##########################################################################

    d = {'result': 0}
    return jsonify(**d)

def identify_shape(img_fp):
    return map(int, check_output("identify -format %%Wx%%H %s" % img_fp, shell=True).split('x'))

@app.route('/compose')
def compose():

    stack = request.args.get('stack', type=str)
    first_section = request.args.get('first_section', type=int)
    last_section = request.args.get('last_section', type=int)
    bad_sections = map(int, request.args.getlist('bad_sections'))
    print 'bad_sections', bad_sections

    elastix_output_dir = os.path.join(data_dir, stack+'_elastix_output')
    input_dir = os.path.join(data_dir, stack+'_thumbnail_renamed')
    aligned_dir = os.path.join(data_dir, stack+'_thumbnail_aligned')

    #################################

    suffix = 'thumbnail'
    all_files = dict(sorted([(int(img_fn[:-4].split('_')[1]), img_fn) for img_fn in os.listdir(input_dir) if suffix in img_fn]))
    all_files = dict([(i, all_files[i]) for i in range(first_section, last_section+1) if i in all_files])

    shapes = Parallel(n_jobs=16)(delayed(identify_shape)(os.path.join(input_dir, img_fn)) for img_fn in all_files.values())
    img_shapes_map = dict(zip(all_files.keys(), shapes))
    img_shapes_arr = np.array(img_shapes_map.values())
    largest_sec = img_shapes_map.keys()[np.argmax(img_shapes_arr[:,0] * img_shapes_arr[:,1])]
    print 'largest section is ', largest_sec

    # no parallelism
    t = time.time()
    print 'composing transform...',
    os.system("ssh gcn-20-33.sdsc.edu %(script_dir)s/compose_transform_thumbnail.py %(stack)s %(elastix_output_dir)s %(first_sec)d %(last_sec)d %(largest_sec)d %(bad_secs)s" % \
    {'stack': stack,
    'script_dir': script_dir,
    'elastix_output_dir': elastix_output_dir,
    'bad_secs': '_'.join(map(str, bad_sections)),
    'first_sec': first_section,
    'last_sec': last_section,
    'largest_sec': largest_sec} )
    print 'done in', time.time() - t, 'seconds'

    ########################################################

    t = time.time()
    print 'warping...',

    run_distributed3('%(script_dir)s/warp_crop_IM.py %(stack)s %(input_dir)s %(aligned_dir)s %%(f)d %%(l)d %(suffix)s 0 0 2000 1500' % \
                    {'stack': stack,
                    'script_dir': script_dir,
                    'input_dir': input_dir,
                    'aligned_dir': aligned_dir,
                    'suffix': suffix},
                    first_sec=first_section,
                    last_sec=last_section,
                    take_one_section=False,
                    stdout=open('/tmp/log', 'ab+'),
                    exclude_nodes=exclude_nodes)

    print 'done in', time.time() - t, 'seconds'

    ########################################################

    d = {'result': 0}
    return jsonify(**d)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False)
