#!/usr/bin/env python

import os
import numpy as np
import sys
import cPickle as pickle
import json

sys.path.append(os.environ['REPO_DIR'] + '/utilities/')
from metadata import *

def parameter_file_to_dict(filename):
    d = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.startswith('('):
                tokens = line[1:-2].split(' ')
                key = tokens[0]
                if len(tokens) > 2:
                    value = []
                    for v in tokens[1:]:
                        try:
                            value.append(float(v))
                        except ValueError:
                            value.append(v)
                else:
                    v = tokens[1]
                    try:
                        value = (float(v))
                    except ValueError:
                        value = v
                d[key] = value

        return d

def parse_parameter_file(filepath):

    d = parameter_file_to_dict(filepath)

    rot_rad, x_mm, y_mm = d['TransformParameters']
    center = np.array(d['CenterOfRotationPoint']) / np.array(d['Spacing'])
    # center[1] = d['Size'][1] - center[1]

    xshift = x_mm / d['Spacing'][0]
    yshift = y_mm / d['Spacing'][1]

    R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                  [np.sin(rot_rad), np.cos(rot_rad)]])
    shift = center + (xshift, yshift) - np.dot(R, center)
    T = np.vstack([np.column_stack([R, shift]), [0,0,1]])

    return T

stack = sys.argv[1]
elastix_output_dir = sys.argv[2]
filenames = json.loads(sys.argv[3])[0]['filenames']
anchor_idx = int(sys.argv[4])
output_fn = sys.argv[5]

#################################################

transformation_to_previous_sec = {}

for i in range(1, len(filenames)):

    custom_tf_fn = os.path.join(DATA_DIR, stack, stack+'_custom_transforms', filenames[i] + '_to_' + filenames[i-1], filenames[i] + '_to_' + filenames[i-1] + '_customTransform.txt')
    custom_tf_fn2 = os.path.join(DATA_DIR, stack, stack+'_custom_transforms', filenames[i] + '_to_' + filenames[i-1], 'TransformParameters.0.txt')
    if os.path.exists(custom_tf_fn):
        # if custom transform is provided
        sys.stderr.write('Load custom transform: %s\n' % custom_tf_fn)
        with open(custom_tf_fn, 'r') as f:
            t11, t12, t13, t21, t22, t23 = map(float, f.readline().split())
        # transformation_to_previous_sec[i] = np.array([[t11, t12, t13], [t21, t22, t23], [0,0,1]])
        transformation_to_previous_sec[i] = np.linalg.inv(np.array([[t11, t12, t13], [t21, t22, t23], [0,0,1]]))
    elif os.path.exists(custom_tf_fn2):
        sys.stderr.write('Load custom transform: %s\n' % custom_tf_fn2)
        transformation_to_previous_sec[i] = parse_parameter_file(custom_tf_fn2)
    else:
        # otherwise, load elastix output
        sys.stderr.write('Load elastix-computed transform: %s\n' % custom_tf_fn2)
        param_fn = os.path.join(elastix_output_dir, filenames[i] + '_to_' + filenames[i-1], 'TransformParameters.0.txt')
        if not os.path.exists(param_fn):
            raise Exception('Transform file does not exist: %s to %s, %s' % (filenames[i], filenames[i-1], param_fn))
        transformation_to_previous_sec[i] = parse_parameter_file(param_fn)
    
    sys.stderr.write('%s\n' % transformation_to_previous_sec[i])

#################################################

transformation_to_anchor_sec = {}

for moving_idx in range(len(filenames)):

    if moving_idx == anchor_idx:
        # transformation_to_anchor_sec[moving_idx] = np.eye(3)
        transformation_to_anchor_sec[filenames[moving_idx]] = np.eye(3)

    elif moving_idx < anchor_idx:
        T_composed = np.eye(3)
        for i in range(anchor_idx, moving_idx, -1):
            T_composed = np.dot(np.linalg.inv(transformation_to_previous_sec[i]), T_composed)
        # transformation_to_anchor_sec[moving_idx] = T_composed
        transformation_to_anchor_sec[filenames[moving_idx]] = T_composed

    else:
        T_composed = np.eye(3)
        for i in range(anchor_idx+1, moving_idx+1):
            T_composed = np.dot(transformation_to_previous_sec[i], T_composed)
        # transformation_to_anchor_sec[moving_idx] = T_composed
        transformation_to_anchor_sec[filenames[moving_idx]] = T_composed

#################################################

with open(output_fn, 'w') as f:
    pickle.dump(transformation_to_anchor_sec, f) 
    # Note that the index starts at 0, BUT the .._renamed folder index starts at 1.