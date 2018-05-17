#! /usr/bin/env python

import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *

from data_manager import *
from metadata import *

from annotation_utilities import *
from registration_utilities import *
from conversion import *

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='')

parser.add_argument("brain_name", type=str, help="Brain name")
parser.add_argument("xml_filepath", type=str, help="Path to Neurolucida exported XML file")
args = parser.parse_args()

brain_name = args.brain_name
xml_filepath = args.xml_filepath

print 'brain_name', brain_name
print 'xml_filepath', xml_filepath

tree = ET.parse(xml_filepath)
root = tree.getroot()

neurolucida_name_map = \
{"Contour Name 1": 'RMC_L',
"Contour Name 2": '3N_L',
# "Contour Name 3": 'fr',
'Brain': 'outline',
 'Brain Outline': 'outline',
 '3N': '3N_L',
 'RN': 'RMC_L',
 'RedNuc': 'RMC_L',
 'SNR': 'SNR_L'}

prefix = '{http://www.mbfbioscience.com/2007/neurolucida}'

contours = defaultdict(list)

# Contour and marker coordinates are in um already.

for item in root.findall(prefix+'contour'):
    name = item.attrib['name']
    if name not in neurolucida_name_map:
        sys.stderr.write('Name %s in stack %s not recognized. Ignored.\n' % (name, brain_name))
        continue
    name = neurolucida_name_map[name]
    curr_contour = []
#     try:
#         resolution = float(item.findall(prefix+'resolution')[0].text)
#         print resolution
#     except:
#         pass
    for p in item.findall(prefix+'point'):
        curr_contour.append((float(p.attrib['x']), float(p.attrib['y']), float(p.attrib['z'])))
    contours[name].append(np.array(curr_contour))

contours.default_factory = None

markers = {}
name = 'All'

for item in root.findall(prefix+'marker'):
    curr_markers = []
    for p in item.findall(prefix+'point'):
        curr_markers.append((float(p.attrib['x']), float(p.attrib['y']), float(p.attrib['z'])))
    markers[name] = np.array(curr_markers)


structure_subset = \
[name for name in contours.keys() if parse_label(name)[0] in all_known_structures]
print structure_subset

out_resolution = '10.0um'
out_resolution_um = convert_resolution_string_to_um(resolution=out_resolution)

markers_orientationCorrected = {name_u: mkrs3d[:, [2,1,0]]*[1,-1,1] for name_u, mkrs3d in markers.iteritems()}
markers_atlasResol = {name: mkrs3d / out_resolution_um for name, mkrs3d in markers_orientationCorrected.iteritems()}

contours_orientationCorrected = {convert_to_left_name(name_u): [cnt[:, [2,1,0]]*[1,-1,1]
                                 for cnt in cnts3d] 
                       for name_u, cnts3d in contours.iteritems()}

contours_atlasResol = {name: [cnt / out_resolution_um
                                for cnt in cnts3d if len(cnt) > 3] 
                       for name, cnts3d in contours_orientationCorrected.iteritems()}
#                                             if name in structure_subset}


# Convert contours to volumes

valid_level = .5

surround_distance_um = 200.
surround_distance_voxel = surround_distance_um / out_resolution_um
print "surround size (in voxels):", surround_distance_voxel

# Reconstruct brain.

reconstructed_brain = {}

for name, cnts3d in contours_atlasResol.iteritems():
    reconstructed_brain[name] = interpolate_contours_to_volume(interpolation_direction='x',
                                                    contours_xyz=cnts3d, 
                                                    len_interval=20.,
                                                        return_origin_instead_of_bbox=True)

    surround_name = convert_to_surround_name(name, margin='%dum' % surround_distance_um)

    reconstructed_brain[surround_name] = \
    get_surround_volume_v2(vol=reconstructed_brain[name][0], origin=reconstructed_brain[name][1], 
                           wall_level=valid_level, distance=surround_distance_voxel, 
                           prob=True,
                           return_origin_instead_of_bbox=True)


for s, v in reconstructed_brain.iteritems():
    vol_fp = DataManager.get_original_volume_filepath_v2(stack_spec=dict(name=brain_name, 
                                                                     vol_type='annotationAsScore',
                                                            resolution=out_resolution),

                                        structure=s)
    save_data(v[0], vol_fp)

    origin_fp = DataManager.get_original_volume_origin_filepath_v3(stack_spec=dict(name=brain_name, 
                                                                     vol_type='annotationAsScore',
                                                                    resolution=out_resolution),
                                        structure=s)
    save_data(v[1], origin_fp)

# Export markers.

for name, mkrs in markers_atlasResol.iteritems():
    save_data(markers_atlasResol[name], 
              DataManager.get_lauren_markers_filepath(stack=brain_name, structure=name, resolution='10.0um'))