#! /usr/bin/env python

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Warp and crop images.')

parser.add_argument("--input_spec", type=str, help="json")
parser.add_argument("--out_prep_id", type=str, help="if not specified, assume None")
parser.add_argument("--input_fp", type=str, help="input filepath")
parser.add_argument("--output_fp", type=str, help="output filepath")
parser.add_argument("--warp", type=str, help="pkl for multiple images or csv_str for one image")
parser.add_argument("--inverse_warp", type=str, help="pkl for multiple images or csv_str for one image")
parser.add_argument("--crop", type=str, help="json or box_xywh_str")
parser.add_argument("--pad_color", type=str, help="background color (black or white)", default='auto')
parser.add_argument("--njobs", type=int, help="Number of parallel jobs", default=1)
parser.add_argument("-r", "--init_rotate", type=str, help="escaped imagemagick convert option string for initial flipping and rotation", default='')
args = parser.parse_args()

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from data_manager import *
from distributed_utilities import *
from metadata import orientation_argparse_str_to_imagemagick_str

def convert_cropbox_from_arr_1um(data, out_fmt, out_resol, stack=None):
    
    data = data / convert_resolution_string_to_um(stack=stack, resolution=out_resol)

    if out_fmt == 'str':
        return ','.join(map(str, data))
    elif out_fmt == 'dict':
        raise Exception("too lazy to implement")
    else:
        return data
    
def convert_cropbox_to_arr_1um(data, in_resol, stack=None):

    if isinstance(data, dict):
        arr = np.array([data['rostral_limit'], data['dorsal_limit'], data['caudal_limit'] - data['rostral_limit'] + 1, data['ventral_limit'] - data['dorsal_limit'] + 1])
        # Since this does not check for wrt, the user needs to make sure the cropbox is relative to the input prep (i.e. the wrt attribute is the same as input prep)
    elif isinstance(data, str):
        arr = np.array(map(int, map(eval, data.split(','))))
    else:
        arr = np.array(data)

    arr_1um = arr * convert_resolution_string_to_um(stack=stack, resolution=in_resol)
    return arr_1um

def convert_cropbox_fmt(out_fmt, data, in_resol='1um', out_resol='1um', stack=None):
    arr_1um = convert_cropbox_to_arr_1um(data=data, in_resol=in_resol, stack=stack)
    return convert_cropbox_from_arr_1um(data=arr_1um, out_fmt=out_fmt, out_resol=out_resol, stack=stack)

def transform_to_str(transform):
    return ','.join(map(str, transform.flatten()))

def str_to_transform(transform_str):
    return np.reshape(map(np.float, transform_str.split(',')), (3,3))

out_prep_id = args.out_prep_id
if out_prep_id == 'None':
    out_prep_id = None 

init_rotate = args.init_rotate
pad_color = args.pad_color

if args.input_spec is not None:
    input_spec = load_ini(args.input_spec)
elif args.input_fp is not None and args.output_fp is not None:
    input_fp = args.input_fp
    output_fp = args.output_fp

if args.crop is not None:
    if args.crop.endswith('.json'):
        cropbox = load_json(args.crop)
        cropbox_resol = cropbox['resolution']
    elif isinstance(args.crop, str):
        cropbox_resol = 'thumbnail'
    
def rescale_transform(transform, factor):

    T = transform.copy()
    T[:2,2] = transform[:2, 2] * factor
    return T

if args.input_spec is not None:

    image_name_list = input_spec['image_name_list']
    stack = input_spec['stack']
    prep_id = input_spec['prep_id']
    if prep_id == 'None': prep_id = None
    version = input_spec['version']
    resol = input_spec['resol']

    if args.warp is not None:
	assert args.warp.endswith('.csv')
	transforms_to_anchor = csv_to_dict(args.warp)
        transforms_to_anchor = {img_name: np.reshape(tf, (3,3)) for img_name, tf in transforms_to_anchor.iteritems()}
    elif args.inverse_warp is not None:
	assert args.inverse_warp.endswith('.csv')
        transforms_to_anchor = csv_to_dict(args.inverse_warp)
        transforms_to_anchor = {img_name: np.linalg.inv(np.reshape(tf, (3,3))) for img_name, tf in transforms_to_anchor.iteritems()}
    else:
	raise ("Must specify either warp or inverse_warp")

    if args.crop is not None:
        if args.crop.endswith('.json'):
	    cropbox = load_json(args.crop)
	    cropboxes = {img_name: cropbox for img_name in image_name_list}
	elif args.crop.endswith('.csv'):
	    cropboxes = csv_to_dict(args.crop)
        elif isinstance(args.crop, str):
            cropboxes = {img_name: args.crop for img_name in image_name_list}
        else:
            raise Exception("crop argument must be either a csv file or a str")

    transforms_resol = 'thumbnail'

    transforms_scale_factor = convert_resolution_string_to_um(stack=stack, resolution=transforms_resol) / convert_resolution_string_to_um(stack=stack, resolution=resol)

    run_distributed('%(script)s --input_fp \"%%(input_fp)s\" --output_fp \"%%(output_fp)s\" %%(transform_str)s %%(box_xywh_str)s --pad_color %%(pad_color)s' % \
                {'script': os.path.join(REPO_DIR, 'preprocess', 'warp_crop.py'),
                },
                kwargs_list=[{'transform_str': ('--warp ' + transform_to_str(rescale_transform(transforms_to_anchor[img_name], factor=transforms_scale_factor))) if args.warp is not None or args.inverse_warp is not None else '',
                              'box_xywh_str': ('--crop ' + convert_cropbox_fmt(data=cropboxes[img_name], out_fmt='str', in_resol=cropbox_resol, out_resol=resol, stack=stack)) if args.crop is not None else '',
                            'input_fp': DataManager.get_image_filepath_v2(stack=stack, fn=img_name, prep_id=prep_id, version=version, resol=resol),
                            'output_fp': DataManager.get_image_filepath_v2(stack=stack, fn=img_name, prep_id=out_prep_id, version=version, resol=resol),
                            'pad_color': ('black' if img_name.split('-')[1][0] == 'F' else 'white') if pad_color == 'auto' else pad_color}
                            for img_name in image_name_list],
                argument_type='single',
               jobs_per_node=args.njobs,
            local_only=True)

else:

    if init_rotate == '':
        init_rotate = ''
    else:
        init_rotate = orientation_argparse_str_to_imagemagick_str[init_rotate]

    if args.warp is not None:
        T = np.linalg.inv(str_to_transform(args.warp))

    if args.crop is not None:
        x, y, w, h = convert_cropbox_fmt(data=args.crop, out_fmt='arr')
    else:
	x, y, w, h = (0,0,2000,1000)

    create_parent_dir_if_not_exists(output_fp)

    execute_command("convert \"%(input_fp)s\" %(init_rotate)s +repage -virtual-pixel background -background %(bg_color)s %(warp_str)s %(crop_str)s -flatten -compress lzw \"%(output_fp)s\"" % \
                {'init_rotate':init_rotate,

'warp_str': ("+distort AffineProjection '%(sx)f,%(rx)f,%(ry)f,%(sy)f,%(tx)f,%(ty)f'" % {
                    'sx':T[0,0],
     'sy':T[1,1],
     'rx':T[1,0],
    'ry':T[0,1],
     'tx':T[0,2],
     'ty':T[1,2],
}) if args.warp is not None else '',
     'input_fp': input_fp,
     'output_fp': output_fp,
      
     'crop_str': '-crop %(w)sx%(h)s%(x)s%(y)s\!' % {
     'x': '+' + str(x) if int(x) >= 0 else str(x),
     'y': '+' + str(y) if int(y) >= 0 else str(y),
     'w': str(w),
     'h': str(h)
     },
     'bg_color': pad_color
}
)
