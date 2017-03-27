#! /usr/bin/env python

from flask import Flask, jsonify, request

app = Flask(__name__)

import os
import argparse
import sys
import time

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
from distributed_utilities import *
from preprocess_utilities import *

script_dir = os.path.join(os.environ['REPO_DIR'], 'preprocess')

exclude_nodes = [33, 47]

@app.route('/')
def index():
    return "Brainstem Web Service"

@app.route('/align')
def align():

    t = time.time()
    print 'aligning...',

    stack = request.args.get('stack', type=str)
    filenames = map(str, request.args.getlist('filenames'))


    run_distributed("%(script_dir)s/align_consecutive_v2.py %(stack)s %(input_dir)s %(elastix_output_dir)s \'%%(kwargs_str)s\'" % \
                    {'stack': stack,
                    'script_dir': script_dir,
                    'input_dir': os.path.join(RAW_DATA_DIR, stack),
                    'elastix_output_dir': os.path.join(DATA_DIR, stack, stack+'_elastix_output')},
                    kwargs_list=[{'prev_fn': filenames[i-1], 'curr_fn': filenames[i]} for i in range(1, len(filenames))],
                    exclude_nodes=exclude_nodes,
                    argument_type='list')

    print 'done in', time.time() - t, 'seconds'

    d = {'result': 0}
    return jsonify(**d)

    ##########################################################################

@app.route('/compose')
def compose():

    stack = request.args.get('stack', type=str)
    filenames = map(str, request.args.getlist('filenames'))
    anchor_fn = request.args.get('anchor_fn', type=str)
    tb_fmt = request.args.get('tb_fmt', type=str)
    pad_bg_color = request.args.get('pad_bg_color', type=str, default='auto')

    elastix_output_dir = os.path.join(DATA_DIR, stack, stack+'_elastix_output')

    #################################

    # no parallelism

    t = time.time()
    print 'composing transform...',

    output_fn = os.path.join(elastix_output_dir, '%(stack)s_transformsTo_%(anchor_fn)s.pkl' % \
                                                    dict(stack=stack, anchor_fn=anchor_fn))

    run_distributed("%(script_dir)s/compose_transform_thumbnail_v2.py %(stack)s %(elastix_output_dir)s \'%%(kwargs_str)s\' %(anchor_idx)d %(output_fn)s" % \
                {'stack': stack,
                'script_dir': script_dir,
                'elastix_output_dir': os.path.join(DATA_DIR, stack, stack+'_elastix_output'),
                'anchor_idx': filenames.index(anchor_fn),
                'output_fn': output_fn},
                kwargs_list=[{'filenames': filenames}],
                use_nodes=[34],
                argument_type='list')

    # transforms_filename = os.path.join(elastix_output_dir, '%(stack)s_transformsTo_%(anchor_fn)s.pkl' % \
    # 														dict(stack=stack, anchor_fn=anchor_fn))
    transforms_filename = '%(stack)s_transformsTo_%(anchor_fn)s.pkl' % dict(stack=stack, anchor_fn=anchor_fn)
    linked_name = os.path.join(elastix_output_dir, '%(stack)s_transformsTo_anchor.pkl' % dict(stack=stack))
    execute_command('rm ' + linked_name)
    execute_command('ln -s ' + transforms_filename + ' ' + linked_name)

    print 'done in', time.time() - t, 'seconds'

    ########################################################

    t = time.time()
    print 'warping...',

    transforms_filename = os.path.join(elastix_output_dir,
                        '%(stack)s_transformsTo_%(anchor_fn)s.pkl' % \
                        dict(stack=stack, anchor_fn=anchor_fn))
    transforms_to_anchor = pickle.load(open(transforms_filename, 'r'))

    if pad_bg_color == 'auto':
        run_distributed('%(script_dir)s/warp_crop_IM_v2.py %(stack)s %(input_dir)s %(aligned_dir)s %%(transform)s %%(filename)s %%(output_fn)s thumbnail 0 0 2000 1500 %%(pad_bg_color)s' % \
                        {'script_dir': script_dir,
                        'stack': stack,
                        'input_dir': os.path.join(RAW_DATA_DIR, stack),
                        'aligned_dir': os.path.join(DATA_DIR, stack, stack + '_thumbnails_alignedTo_' + anchor_fn)
                        },
                        kwargs_list=[{'transform': ','.join(map(str, transforms_to_anchor[fn].flatten())),
                                    'filename': fn + '.' + tb_fmt,
                                    'output_fn': fn + '_thumbnail_alignedTo_' + anchor_fn + '.tif',
                                    'pad_bg_color': 'black' if fn.split('-')[1][0] == 'F' else 'white'}
                                    for fn in filenames],
                        exclude_nodes=exclude_nodes + [32],
                        argument_type='single')
    else:
        run_distributed('%(script_dir)s/warp_crop_IM_v2.py %(stack)s %(input_dir)s %(aligned_dir)s %%(transform)s %%(filename)s %%(output_fn)s thumbnail 0 0 2000 1500 %(pad_bg_color)s' % \
                        {'script_dir': script_dir,
                        'stack': stack,
                        'input_dir': os.path.join(RAW_DATA_DIR, stack),
                        'aligned_dir': os.path.join(DATA_DIR, stack, stack + '_thumbnails_alignedTo_' + anchor_fn),
                        'pad_bg_color': pad_bg_color},
                        kwargs_list=[{'transform': ','.join(map(str, transforms_to_anchor[fn].flatten())),
                                    'filename': fn + '.' + tb_fmt,
                                    'output_fn': fn + '_thumbnail_alignedTo_' + anchor_fn + '.tif'}
                                    for fn in filenames],
                        exclude_nodes=exclude_nodes + [32],
                        argument_type='single')

    print 'done in', time.time() - t, 'seconds'

    ########################################################

    d = {'result': 0}
    return jsonify(**d)


@app.route('/crop')
def crop():

    stack = request.args.get('stack', type=str)
    filenames = map(str, request.args.getlist('filenames'))
    x = request.args.get('x', type=int)
    y = request.args.get('y', type=int)
    w = request.args.get('w', type=int)
    h = request.args.get('h', type=int)
    first_fn = request.args.get('first_fn', type=str)
    last_fn = request.args.get('last_fn', type=str)
    anchor_fn = request.args.get('anchor_fn', type=str)
    pad_bg_color = request.args.get('pad_bg_color', type=str)

    first_idx = filenames.index(first_fn)
    last_idx = filenames.index(last_fn)

    ##################################################

    t = time.time()
    sys.stderr.write('cropping thumbnail...')

    os.system(('mkdir %(stack_data_dir)s/%(stack)s_thumbnails_alignedTo_%(anchor_fn)s_cropped; '
                'mogrify -set filename:name %%t -crop %(w)dx%(h)d+%(x)d+%(y)d -write "%(stack_data_dir)s/%(stack)s_thumbnails_alignedTo_%(anchor_fn)s_cropped/%%[filename:name]_cropped.tif" %(stack_data_dir)s/%(stack)s_thumbnails_alignedTo_%(anchor_fn)s/*.tif') % \
    	{'stack': stack,
    	'stack_data_dir': os.path.join(DATA_DIR, stack),
    	'w':w, 'h':h, 'x':x, 'y':y,
        'anchor_fn': anchor_fn})

    sys.stderr.write('done in %f seconds\n' % (time.time() - t))

    ################################################

    t = time.time()
    sys.stderr.write('expanding...')

    expanded_tif_dir = create_if_not_exists(os.environ['DATA_DIR'] + '/' + stack + '/' + stack + '_lossless_tif')
    # jp2_dir = os.environ['DATA_DIR'] + '/' + stack + '_lossless_renamed_jp2'

    filenames_to_expand = [fn for fn in filenames[first_idx:last_idx+1] if not os.path.exists(expanded_tif_dir + '/' + fn + '_lossless.tif')]
    sys.stderr.write('filenames_to_expand: %s' % filenames_to_expand)

    run_distributed('kdu_expand_patched -i %(jp2_dir)s/%%(fn)s_lossless.jp2 -o %(expanded_tif_dir)s/%%(fn)s_lossless.tif' % \
                    {'jp2_dir': '/home/yuncong/CSHL_data/' + stack,
                    # 'stack': stack,
                    'expanded_tif_dir': expanded_tif_dir},
                    kwargs_list={'fn': filenames_to_expand},
                    exclude_nodes=exclude_nodes,
                    argument_type='single')

    sys.stderr.write('done in %f seconds\n' % (time.time() - t))

    ################################################

    t = time.time()
    sys.stderr.write('warping and cropping lossless...')

    elastix_output_dir = os.path.join(DATA_DIR, stack, stack+'_elastix_output')
    transforms_to_anchor = pickle.load(open(elastix_output_dir + '/%(stack)s_transformsTo_anchor.pkl' % {'stack':stack}, 'r'))
    # Note that the index from trasform pickle file starts at 0, BUT the .._renamed folder index starts at 1.#

    # print transforms_to_anchor.keys()

    if pad_bg_color == 'auto':
        # If alternating, then black padding for F sections, white padding for N sections.
        run_distributed(command='%(script_path)s %(stack)s %(lossless_tif_dir)s %(lossless_aligned_cropped_dir)s %%(transform)s %%(filename)s %%(output_fn)s lossless %(x)d %(y)d %(w)d %(h)d %%(pad_bg_color)s'%\
                        {'script_path': script_dir + '/warp_crop_IM_v2.py',
                        'stack': stack,
                        'lossless_tif_dir': os.path.join(os.environ['DATA_DIR'] , stack, stack + '_lossless_tif'),
                        'lossless_aligned_cropped_dir': os.path.join( os.environ['DATA_DIR'], stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped'),
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h},
                        kwargs_list=[{'transform': ','.join(map(str, transforms_to_anchor[fn].flatten())),
                                    'filename': fn + '_lossless.tif',
                                    'output_fn': fn + '_lossless_alignedTo_' + anchor_fn + '_cropped.tif',
                                    'pad_bg_color': 'black' if fn.split('-')[1][0] == 'F' else 'white'}
                                    for fn in filenames[first_idx:last_idx+1]],
                        exclude_nodes=exclude_nodes + [32], # "convert" command is especially slow on gcn-20-32 for some reason.
                        argument_type='single')
    else:
        run_distributed(command='%(script_path)s %(stack)s %(lossless_tif_dir)s %(lossless_aligned_cropped_dir)s %%(transform)s %%(filename)s %%(output_fn)s lossless %(x)d %(y)d %(w)d %(h)d %(pad_bg_color)s'%\
                        {'script_path': script_dir + '/warp_crop_IM_v2.py',
                        'stack': stack,
                        'lossless_tif_dir': os.path.join(os.environ['DATA_DIR'] , stack, stack + '_lossless_tif'),
                        'lossless_aligned_cropped_dir': os.path.join( os.environ['DATA_DIR'], stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped'),
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'pad_bg_color': pad_bg_color},
                        kwargs_list=[{'transform': ','.join(map(str, transforms_to_anchor[fn].flatten())),
                                    'filename': fn + '_lossless.tif',
                                    'output_fn': fn + '_lossless_alignedTo_' + anchor_fn + '_cropped.tif'}
                                    for fn in filenames[first_idx:last_idx+1]],
                        exclude_nodes=exclude_nodes + [32], # "convert" command is especially slow on gcn-20-32 for some reason.
                        argument_type='single')

    sys.stderr.write('done in %f seconds\n' % (time.time() - t))

    #########################################

    # if stack in all_ntb_stacks:
    #
    #     # Do gray for NT
    #     t = time.time()
    #     print 'Neurotrace to grayscale...',
    #
    #     input_dir = os.path.join( DATA_DIR, stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped')
    #     output_dir = create_if_not_exists(os.path.join(DATA_DIR, stack, '%(stack)s_lossless_alignedTo_%(anchor_fn)s_cropped_blueAsGrayscale' % dict(stack=stack, anchor_fn=anchor_fn)))
    #
    #     run_distributed4(command='%(script_path)s %(input_dir)s %(output_dir)s %%(filename)s' % \
    #                     {'script_path': os.path.join(os.environ['REPO_DIR'], 'preprocess', 'neurotrace_blue_to_nissl.py'),
    #                     'input_dir': input_dir,
    #                     'output_dir': output_dir},
    #                     kwargs_list=dict(filename=[f + '_lossless_alignedTo_' + anchor_fn + '_cropped.tif' for f in filenames[first_idx:last_idx+1]]),
    #                     exclude_nodes=exclude_nodes,
    #                     argument_type='single')
    #
    #     print 'done in', time.time() - t, 'seconds'
    #
    #     # Do compress for NT
    #     t = time.time()
    #     print 'Generating compressed version and/or saturation as grayscale...',
    #
    #     run_distributed4('%(script_dir)s/generate_other_versions_v2.py %(stack)s %(input_dir)s \'%%(input_filenames)s\' --output_compressed_dir %(compressed_dir)s' % \
    #                     dict(script_dir=script_dir,
    #                     stack=stack,
    #                     input_dir=os.path.join(data_dir, stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped'),
    #                     compressed_dir=os.path.join(data_dir, stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped_compressed')),
    #                     kwargs_list={'input_filenames': [fn + '_lossless_alignedTo_' + anchor_fn + '_cropped_blueAsGrayscale.tif' for fn in filenames[first_idx:last_idx+1]]},
    #                     exclude_nodes=exclude_nodes + [32],
    #                     argument_type='list2')
    #
    #     print 'done in', time.time() - t, 'seconds'
    #
    # elif stack in all_nissl_stacks:
    #
    #     # Do comp. and sat. for Nissl
    #     run_distributed4('%(script_dir)s/generate_other_versions_v2.py %(stack)s %(input_dir)s \'%%(input_filenames)s\' --output_compressed_dir %(compressed_dir)s --output_saturation_dir %(saturation_dir)s' % \
    #                     dict(script_dir=script_dir,
    #                     stack=stack,
    #                     input_dir=os.path.join(data_dir, stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped'),
    #                     compressed_dir=os.path.join(data_dir, stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped_compressed'),
    #                     saturation_dir=os.path.join(data_dir, stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped_saturation')),
    #                     kwargs_list={'input_filenames': [fn + '_lossless_alignedTo_' + anchor_fn + '_cropped.tif' for fn in filenames[first_idx:last_idx+1]]},
    #                     exclude_nodes=exclude_nodes + [32],
    #                     argument_type='list2')
    #
    # elif stack in all_alt_nissl_ntb_stacks:
    #
    #     # Do gray for NT
    #     print 'Neurotrace to grayscale...',
    #     input_dir = os.path.join( DATA_DIR, stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped')
    #     output_dir = create_if_not_exists(os.path.join(DATA_DIR, stack, '%(stack)s_lossless_alignedTo_%(anchor_fn)s_cropped_blueAsGrayscale' % dict(stack=stack, anchor_fn=anchor_fn)))
    #     run_distributed4(command='%(script_path)s %(input_dir)s %(output_dir)s %%(filename)s' % \
    #                     {'script_path': os.path.join(os.environ['REPO_DIR'], 'preprocess', 'neurotrace_blue_to_nissl.py'),
    #                     'input_dir': input_dir,
    #                     'output_dir': output_dir},
    #                     kwargs_list=dict(filename=[f + '_lossless_alignedTo_' + anchor_fn + '_cropped.tif' for f in filenames[first_idx:last_idx+1] \
    #                                                 if f.split('-')[1][0] == 'F']),
    #                     exclude_nodes=exclude_nodes,
    #                     argument_type='single')
    #
    #     # Do comp. for NT
    #     run_distributed4('%(script_dir)s/generate_other_versions_v2.py %(stack)s %(input_dir)s \'%%(input_filenames)s\' --output_compressed_dir %(compressed_dir)s' % \
    #                     dict(script_dir=script_dir,
    #                     stack=stack,
    #                     input_dir=os.path.join(data_dir, stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped'),
    #                     compressed_dir=os.path.join(data_dir, stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped_compressed')),
    #                     kwargs_list={'input_filenames': [fn + '_lossless_alignedTo_' + anchor_fn + '_cropped_blueAsGrayscale.tif' for fn in filenames[first_idx:last_idx+1] \
    #                                                     if fn.split('-')[1][0] == 'F']},
    #                     exclude_nodes=exclude_nodes + [32],
    #                     argument_type='list2')
    #
    #     # Do comp. and sat. for Nissl
    #     run_distributed4('%(script_dir)s/generate_other_versions_v2.py %(stack)s %(input_dir)s \'%%(input_filenames)s\' --output_compressed_dir %(compressed_dir)s --output_saturation_dir %(saturation_dir)s' % \
    #                     dict(script_dir=script_dir,
    #                     stack=stack,
    #                     input_dir=os.path.join(data_dir, stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped'),
    #                     compressed_dir=os.path.join(data_dir, stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped_compressed'),
    #                     saturation_dir=os.path.join(data_dir, stack, stack + '_lossless_alignedTo_' + anchor_fn + '_cropped_saturation')),
    #                     kwargs_list={'input_filenames': [fn + '_lossless_alignedTo_' + anchor_fn + '_cropped.tif' for fn in filenames[first_idx:last_idx+1]
    #                                                     if fn.split('-')[1][0] == 'N']},
    #                     exclude_nodes=exclude_nodes + [32],
    #                     argument_type='list2')
    # else:
    #     raise

    #########################################

    d = {'result': 0}
    return jsonify(**d)


# @app.route('/confirm_order')
# def confirm_order():
#
#     stack = request.args.get('stack', type=str)
#     sorted_filenames = request.args.getlist('sorted_filenames')
#     anchor_fn = request.args.get('anchor_fn', type=str)
#
#     ###### Generate thumbnail sorted symlinks ######
#
#     cmd = ('cd %(data_dir)s/%(stack)s &&'
#             'rm -rf %(stack)s_thumbnail_sorted_aligned &&'
#             'mkdir %(stack)s_thumbnail_sorted_aligned') % \
#             dict(stack=stack, data_dir=DATA_DIR)
#     execute_command(cmd)
#
#     for idx, fn in enumerate(sorted_filenames):
#
#         if not os.path.exists('%(data_dir)s/%(stack)s/%(stack)s_thumbnail_unsorted_alignedTo_%(anchor_fn)s/%(fn)s_thumbnail_alignedTo_%(anchor_fn)s.tif' % \
#                 dict(stack=stack, fn=fn, anchor_fn=anchor_fn,  data_dir=DATA_DIR)):
#             continue
#
#         cmd = ('cd %(data_dir)s/%(stack)s &&'
#                 'ln -s ../%(stack)s_thumbnail_unsorted_alignedTo_%(anchor_fn)s/%(fn)s_thumbnail_alignedTo_%(anchor_fn)s.tif '
#                 '%(stack)s_thumbnail_sorted_aligned/%(stack)s_%(idx)04d_thumbnail_aligned.tif') % \
#                 dict(stack=stack, fn=fn, idx=idx+1, anchor_fn=anchor_fn, data_dir=DATA_DIR)
#         execute_command(cmd)
#
#
#     ###### Generate thumbnail cropped sorted symlinks ######
#
#     cmd = ('cd %(data_dir)s/%(stack)s &&'
#             'rm -rf %(stack)s_thumbnail_sorted_aligned_cropped &&'
#             'mkdir %(stack)s_thumbnail_sorted_aligned_cropped') % \
#             dict(stack=stack, data_dir=DATA_DIR)
#     execute_command(cmd)
#
#     for idx, fn in enumerate(sorted_filenames):
#
#         if not os.path.exists('%(data_dir)s/%(stack)s/%(stack)s_thumbnail_unsorted_alignedTo_%(anchor_fn)s_cropped/%(fn)s_thumbnail_alignedTo_%(anchor_fn)s_cropped.tif' % \
#                 dict(stack=stack, fn=fn, anchor_fn=anchor_fn, data_dir=DATA_DIR)):
#             continue
#
#         cmd = ('cd %(data_dir)s/%(stack)s &&'
#                 'ln -s ../%(stack)s_thumbnail_unsorted_alignedTo_%(anchor_fn)s_cropped/%(fn)s_thumbnail_alignedTo_%(anchor_fn)s_cropped.tif '
#                 '%(stack)s_thumbnail_sorted_aligned_cropped/%(stack)s_%(idx)04d_thumbnail_aligned_cropped.tif') % \
#                 dict(stack=stack, fn=fn, idx=idx+1, anchor_fn=anchor_fn, data_dir=DATA_DIR)
#         execute_command(cmd)
#
#
#     ###### Generate lossless sorted symlinks ######
#
#     cmd = ('cd %(data_dir)s/%(stack)s &&'
#             'rm -rf %(stack)s_lossless_sorted_aligned_cropped &&'
#             'mkdir %(stack)s_lossless_sorted_aligned_cropped') % \
#             dict(stack=stack, data_dir=DATA_DIR)
#     execute_command(cmd)
#
#     for idx, fn in enumerate(sorted_filenames):
#
#         if not os.path.exists('%(data_dir)s/%(stack)s/%(stack)s_lossless_unsorted_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped.tif' % \
#                 dict(stack=stack, fn=fn, anchor_fn=anchor_fn, data_dir=DATA_DIR)):
#             continue
#
#         cmd = ('cd %(data_dir)s/%(stack)s &&'
#                 'ln -s ../%(stack)s_lossless_unsorted_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped.tif '
#                 '%(stack)s_lossless_sorted_aligned_cropped/%(stack)s_%(idx)04d_lossless_aligned_cropped.tif') % \
#                 dict(stack=stack, fn=fn, idx=idx+1, anchor_fn=anchor_fn, data_dir=DATA_DIR)
#         execute_command(cmd)
#
#     ###### Generate compressed sorted symlinks ######
#
#     cmd = ('cd %(data_dir)s/%(stack)s &&'
#             'rm -rf %(stack)s_lossless_sorted_aligned_cropped_compressed &&'
#             'mkdir %(stack)s_lossless_sorted_aligned_cropped_compressed') % \
#             dict(stack=stack, data_dir=DATA_DIR)
#     execute_command(cmd)
#
#     for idx, fn in enumerate(sorted_filenames):
#
#         if not os.path.exists('%(data_dir)s/%(stack)s/%(stack)s_lossless_unsorted_alignedTo_%(anchor_fn)s_cropped_compressed/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_compressed.jpg' % \
#                 dict(stack=stack, fn=fn, anchor_fn=anchor_fn, data_dir=DATA_DIR)):
#             continue
#
#         cmd = ('cd %(data_dir)s/%(stack)s &&'
#                 'ln -s ../%(stack)s_lossless_unsorted_alignedTo_%(anchor_fn)s_cropped_compressed/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_compressed.jpg '
#                 '%(stack)s_lossless_sorted_aligned_cropped_compressed/%(stack)s_%(idx)04d_lossless_aligned_cropped_compressed.jpg') % \
#                 dict(stack=stack, fn=fn, idx=idx+1, anchor_fn=anchor_fn, data_dir=DATA_DIR)
#         execute_command(cmd)
#
#     ###### Generate saturation sorted symlinks ######
#
#     cmd = ('cd %(data_dir)s/%(stack)s &&'
#             'rm -rf %(stack)s_lossless_sorted_aligned_cropped_saturation &&'
#             'mkdir %(stack)s_lossless_sorted_aligned_cropped_saturation') % \
#             dict(stack=stack, data_dir=DATA_DIR)
#     execute_command(cmd)
#
#     for idx, fn in enumerate(sorted_filenames):
#
#         if not os.path.exists('%(data_dir)s/%(stack)s/%(stack)s_lossless_unsorted_alignedTo_%(anchor_fn)s_cropped_saturation/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_saturation.jpg' % \
#                 dict(stack=stack, fn=fn, anchor_fn=anchor_fn, data_dir=DATA_DIR)):
#             continue
#
#         cmd = ('cd %(data_dir)s/%(stack)s &&'
#                 'ln -s ../%(stack)s_lossless_unsorted_alignedTo_%(anchor_fn)s_cropped_saturation/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_saturation.jpg '
#                 '%(stack)s_lossless_sorted_aligned_cropped_saturation/%(stack)s_%(idx)04d_lossless_aligned_cropped_saturation.jpg') % \
#                 dict(stack=stack, fn=fn, idx=idx+1, anchor_fn=anchor_fn, data_dir=DATA_DIR)
#         execute_command(cmd)
#
#     ###### Generate thumbnail aligned mask sorted symlinks ######
#
#     cmd = ('cd %(data_dir)s/%(stack)s &&'
#             'rm -rf %(stack)s_mask_sorted_aligned &&'
#             'mkdir %(stack)s_mask_sorted_aligned') % \
#             dict(stack=stack, data_dir=DATA_DIR)
#     execute_command(cmd)
#
#     for idx, fn in enumerate(sorted_filenames):
#
#         if not os.path.exists('%(data_dir)s/%(stack)s/%(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s/%(fn)s_mask_alignedTo_%(anchor_fn)s.png' % \
#                 dict(stack=stack, fn=fn, anchor_fn=anchor_fn, data_dir=DATA_DIR)):
#             continue
#
#         cmd = ('cd %(data_dir)s/%(stack)s &&'
#                 'ln -s ../%(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s/%(fn)s_mask_alignedTo_%(anchor_fn)s.png '
#                 '%(stack)s_mask_sorted_aligned/%(stack)s_%(idx)04d_mask_aligned.png') % \
#                 dict(stack=stack, fn=fn, idx=idx+1, anchor_fn=anchor_fn, data_dir=DATA_DIR)
#         execute_command(cmd)
#
#     ###### Generate thumbnail aligned cropped mask sorted symlinks ######
#
#     cmd = ('cd %(data_dir)s/%(stack)s &&'
#             'rm -rf %(stack)s_mask_sorted_aligned_cropped &&'
#             'mkdir %(stack)s_mask_sorted_aligned_cropped') % \
#             dict(stack=stack, data_dir=DATA_DIR)
#     execute_command(cmd)
#
#     for idx, fn in enumerate(sorted_filenames):
#
#         if not os.path.exists('%(data_dir)s/%(stack)s/%(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s_cropped/%(fn)s_mask_alignedTo_%(anchor_fn)s_cropped.png' % \
#                 dict(stack=stack, fn=fn, anchor_fn=anchor_fn, data_dir=DATA_DIR)):
#             continue
#
#         cmd = ('cd %(data_dir)s/%(stack)s &&'
#                 'ln -s ../%(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s_cropped/%(fn)s_mask_alignedTo_%(anchor_fn)s_cropped.png '
#                 '%(stack)s_mask_sorted_aligned_cropped/%(stack)s_%(idx)04d_mask_aligned_cropped.png') % \
#                 dict(stack=stack, fn=fn, idx=idx+1, anchor_fn=anchor_fn, data_dir=DATA_DIR)
#         execute_command(cmd)
#
#
#     d = {'result': 0}
#     return jsonify(**d)


@app.route('/generate_masks')
def generate_masks():

    args = request.args

    stack = args.get('stack', type=str)
    filenames = map(str, args.getlist('filenames'))
    border_dissim_percentile = args.get('border_dissim_percentile', default=DEFAULT_BORDER_DISSIMILARITY_PERCENTILE, type=int)
    min_size = args.get('min_size', default=DEFAULT_MINSIZE, type=int)
    # output_mode = args.get('output_mode', default='normal', type=str)

    ##################################################

    # t = time.time()
    # print 'Regularize colorspace for neurotrace images...',
    #
    # input_dir = '/home/yuncong/CSHL_data/%(stack)s' % dict(stack=stack)
    # output_dir = create_if_not_exists('/home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_brightfieldized' % dict(stack=stack))
    #
    # run_distributed4(command='%(script_path)s %(input_dir)s %(output_dir)s %%(filename)s' % \
    #                 {'script_path': os.path.join(os.environ['REPO_DIR'], 'preprocess') + '/neurotrace_blue_to_nissl.py',
    #                 'input_dir': input_dir,
    #                 'output_dir': output_dir},
    #                 kwargs_list=dict(filename=[fn + '.' + tb_fmt for fn in filenames \
    #                                             if fn.split('-')[1][0] == 'F']),
    #                 exclude_nodes=exclude_nodes,
    #                 # use_nodes=[35],
    #                 argument_type='single')
    #
    # print 'done in', time.time() - t, 'seconds'


    ##################################################

    t = time.time()
    print 'Generating thumbnail mask...',

    input_dir = '/home/yuncong/CSHL_data/%(stack)s' % dict(stack=stack)
    output_dir = create_if_not_exists(os.path.join(DATA_DIR, stack, stack + '_submasks'))
    # if output_mode == 'normal':
    #     output_dir = create_if_not_exists('/home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_submasks' % dict(stack=stack))
    # elif output_mode == 'alternative':
    #     output_dir = create_if_not_exists('/home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_alternative_submasks' % dict(stack=stack))

    script_name = 'generate_thumbnail_masks_v4.py'
    # !! For some reason (perhaps too much simultaneous write to disk), the distributed computation cannot finish, usually stuck with only a few sections left.

    if 'fg_dissim_thresh' in args:
        fg_dissim_thresh = args.get('fg_dissim_thresh', type=float)

        run_distributed(command='%(script_path)s %(stack)s %(input_dir)s \'%%(filenames)s\' %(output_dir)s --border_dissim_percentile %(border_dissim_percentile)d --fg_dissim_thresh %(fg_dissim_thresh).2f --min_size %(min_size)d' % \
                        {'script_path': os.path.join(os.environ['REPO_DIR'], 'preprocess', script_name),
                        'stack': stack,
                        'input_dir': input_dir,
                        'output_dir': output_dir,
                        'border_dissim_percentile': border_dissim_percentile,
                        'fg_dissim_thresh': fg_dissim_thresh,
                        'min_size': min_size},
                        kwargs_list=dict(filenames=filenames),
                        exclude_nodes=exclude_nodes,
                        argument_type='list2')
    else:
        run_distributed(command='%(script_path)s %(stack)s %(input_dir)s \'%%(filenames)s\' %(output_dir)s --border_dissim_percentile %(border_dissim_percentile)d --min_size %(min_size)d' % \
                        {'script_path': os.path.join(os.environ['REPO_DIR'], 'preprocess', script_name),
                        'stack': stack,
                        'input_dir': input_dir,
                        'output_dir': output_dir,
                        'border_dissim_percentile': border_dissim_percentile,
                        'min_size': min_size},
                        kwargs_list=dict(filenames=filenames),
                        exclude_nodes=exclude_nodes,
                        argument_type='list2')

    print 'done in', time.time() - t, 'seconds' # 1600s

    ##################################################

    # t = time.time()
    # print 'Generating visualization of mask contours overlayed on thumbnail images ...',
    #
    # image_dir = '%(raw_data_dir)s/%(stack)s' % dict(stack=stack, raw_data_dir=RAW_DATA_DIR)
    # # image_dir = '/home/yuncong/CSHL_data_processed/%(stack)s/%(stack)s_brightfieldized' % dict(stack=stack)
    # mask_dir = '%(data_dir)s/%(stack)s/%(stack)s_mask_unsorted' % dict(stack=stack, data_dir=DATA_DIR)
    # output_dir = create_if_not_exists('%(data_dir)s/%(stack)s/%(stack)s_maskContourViz_unsorted' % dict(stack=stack, data_dir=DATA_DIR))
    #
    # run_distributed4(command='%(script_path)s %(stack)s %(image_dir)s %(mask_dir)s \'%%(filenames)s\' %(output_dir)s --tb_fmt %(tb_fmt)s' % \
    #                 {'script_path': os.path.join(os.environ['REPO_DIR'], 'preprocess') + '/generate_thumbnail_mask_contour_viz.py',
    #                 'stack': stack,
    #                 'image_dir': image_dir,
    #                 'mask_dir': mask_dir,
    #                 'output_dir': output_dir,
    #                 'tb_fmt': tb_fmt},
    #                 kwargs_list=dict(filenames=filenames),
    #                 exclude_nodes=exclude_nodes,
    #                 argument_type='list2')

    # print 'done in', time.time() - t, 'seconds'

    # if output_mode == 'alternative':

        # review_results_all_images = {}
        # all_submasks_encoded = defaultdict(dict)
        # for fn in filenames:
        #     submask_dir = '/home/yuncong/csd395/CSHL_data_processed/%(stack)s/%(stack)s_alternative_submasks/%(fn)s' % dict(stack=stack, fn=fn)
        #     review_result_fp = os.path.join(submask_dir, '%(fn)s_submasksAlgReview.txt' % dict(fn=fn))
        #     review_result = read_dict_from_txt(review_result_fp, converter=np.int, key_converter=np.int)
        #     review_results_all_images[fn] = review_result
        #     for submask_ind, decision in review_result.iteritems():
        #         submask_fp = os.path.join(submask_dir, '%(fn)s_submask_%(submask_ind)d.png' % dict(fn=fn, submask_ind=submask_ind))
                # submask = (imread(submask_fp) > 0).astype(np.int)
                # all_submasks_encoded[fn][submask_ind] = bp.pack_ndarray_str(submask)
                # all_submasks_encoded[fn][submask_ind] = submask_fp

        # d = {'result': 0, 'submasks_filepaths': all_submasks_encoded, 'review_decisions': review_results_all_images}
    # else:
        # d = {'result': 0}

    d = {'result': 0}
    return jsonify(**d)

@app.route('/warp_crop_masks')
def warp_crop_masks():

    stack = request.args.get('stack', type=str)
    filenames = map(str, request.args.getlist('filenames'))
    x = request.args.get('x', type=int)
    y = request.args.get('y', type=int)
    w = request.args.get('w', type=int)
    h = request.args.get('h', type=int)
    anchor_fn = request.args.get('anchor_fn', type=str)

    ########################################################

    t = time.time()
    print 'warping thumbnail mask...',

    elastix_output_dir = os.path.join(DATA_DIR, stack, stack+'_elastix_output')
    transforms_filename = os.path.join(elastix_output_dir, '%(stack)s_transformsTo_%(anchor_fn)s.pkl' % \
                                                            dict(stack=stack, anchor_fn=anchor_fn))
    transforms_to_anchor = pickle.load(open(transforms_filename, 'r'))

    execute_command('rm -rf %(aligned_dir)s' % dict(aligned_dir=os.path.join(DATA_DIR, stack, stack + '_masks_alignedTo_' + anchor_fn)))

    run_distributed('%(script_dir)s/warp_crop_IM_v2.py %(stack)s %(input_dir)s %(aligned_dir)s %%(transform)s %%(filename)s %%(output_fn)s thumbnail 0 0 2000 1500 black' % \
                    {'script_dir': script_dir,
                    'stack': stack,
                    'input_dir': os.path.join(DATA_DIR, stack, stack + '_masks'),
                    'aligned_dir': os.path.join(DATA_DIR, stack, stack + '_masks_alignedTo_' + anchor_fn)},
                    kwargs_list=[{'transform': ','.join(map(str, transforms_to_anchor[fn].flatten())),
                                'filename': fn + '_mask.png',
                                'output_fn': fn + '_mask_alignedTo_' + anchor_fn + '.png'}
                                for fn in filenames],
                    exclude_nodes=exclude_nodes + [32],
                    argument_type='single')

    print 'done in', time.time() - t, 'seconds'

    ########################################################

    t = time.time()
    sys.stderr.write('cropping thumbnail mask...')

    aligned_cropped_dir = '%(stack_data_dir)s/%(stack)s_masks_alignedTo_%(anchor_fn)s_cropped' % \
                            {'stack': stack,
                            'stack_data_dir': os.path.join(DATA_DIR, stack),
                            'anchor_fn': anchor_fn}

    execute_command('rm -rf %(aligned_cropped_dir)s' % dict(aligned_cropped_dir=aligned_cropped_dir))

    os.system(('mkdir %(aligned_cropped_dir)s ; '
                'mogrify -set filename:name %%t -crop %(w)dx%(h)d+%(x)d+%(y)d -write "%(aligned_cropped_dir)s/%%[filename:name]_cropped.png" %(stack_data_dir)s/%(stack)s_masks_alignedTo_%(anchor_fn)s/*.png') % \
    	{'stack': stack,
    	'stack_data_dir': os.path.join(DATA_DIR, stack),
        'aligned_cropped_dir': aligned_cropped_dir,
    	'w':w, 'h':h, 'x':x, 'y':y,
        'anchor_fn': anchor_fn})

    sys.stderr.write('done in %f seconds\n' % (time.time() - t))

    d = {'result': 0}
    return jsonify(**d)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False)
