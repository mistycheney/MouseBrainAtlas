#! /usr/bin/env python

import sys
import os
from ast import literal_eval
import time
import argparse

import numpy as np
import bloscpack as bp
import pandas as pd

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from annotation_utilities import *
from registration_utilities import *
from vis3d_utilities import *
from metadata import *
from data_manager import *

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='This script renders the atlas in 3D using VTK, with optional markers.')

# parser.add_argument("--fixed_brain_spec", type=str, help="Fixed brain specification, json", default='demo_fixed_brain_spec_12N.json')
# parser.add_argument("--moving_brain_spec", type=str, help="Moving brain specification, json", default='demo_moving_brain_spec_12N.json')
# parser.add_argument("-r", "--registration_setting", type=int, help="Registration setting, int, defined in registration_settings.csv", default=7)
# parser.add_argument("-g", "--use_simple_global", action='store_false', help="Set this flag to NOT initialize with simple global registration")
# parser.add_argument("--out_dir", type=str, help="Output directory")
parser.add_argument("--experiments_config", type=str, help="A csv file with each row specifying info for one experiment. Default: %(default)s", default='lauren_experiments.csv')

args = parser.parse_args()

# atlas_spec = load_json(args.fixed_brain_spec)
# brain_m_spec = load_json(args.moving_brain_spec)
# registration_setting = args.registration_setting
# use_simple_global = args.use_simple_global

experiments = pd.read_csv(args.experiments_config, index_col=0).T.to_dict()

atlas_name = 'atlasV7'
atlas_spec = dict(name=atlas_name, resolution='10.0um', vol_type='score')

SNR_L_vol_10um, SNR_L_origin_10um_wrt_canonicalAtlasSpace =\
DataManager.load_original_volume_v2(stack_spec=atlas_spec, structure='SNR_L', bbox_wrt='canonicalAtlasSpace')

# SNR_R_vol_10um, SNR_R_ori_10um_wrt_canonicalAtlasSpace =\
# DataManager.load_original_volume_v2(stack_spec=atlas_spec, structure='SNR_R', bbox_wrt='canonicalAtlasSpace')
# SNR_L_nominal_location_1um_wrt_canonicalAtlasSpace = load_data(DataManager.get_structure_mean_positions_filepath(atlas_name=atlas_name, resolution='1um'))['SNR_L']
# SNR_L_nominal_location_10um_wrt_canonicalAtlasSpace = SNR_L_nominal_location_1um_wrt_canonicalAtlasSpace / 10.
# SNR_L_vol_10um, SNR_L_origin_10um_wrt_canonicalAtlasSpace = \
# mirror_volume_v2(SNR_R_vol_10um, SNR_L_nominal_location_10um_wrt_canonicalAtlasSpace)

level = 0.000001
num_simplify_iter = 4

SNR_L_mesh_level01_vertices_10um, SNR_L_mesh_level01_faces = \
volume_to_polydata(volume=(SNR_L_vol_10um, SNR_L_origin_10um_wrt_canonicalAtlasSpace),
                   level=level,
                     num_simplify_iter=num_simplify_iter,
                     smooth=True,
                     return_vertex_face_list=True)

SNR_L_mesh_level01_um = mesh_to_polydata(vertices=SNR_L_mesh_level01_vertices_10um * 10.,
faces=SNR_L_mesh_level01_faces)

####################################

atlas_meshes_10um = DataManager.load_meshes_v2(atlas_spec, sided=True, return_polydata_only=False,
                                               include_surround=False, levels=0.5)
atlas_meshes_um = {s: mesh_to_polydata(vertices=v*10., faces=f) for s, (v, f) in atlas_meshes_10um.iteritems()}
atlas_meshes_um['SNR_L'] = SNR_L_mesh_level01_um

atlas_structure_actors_um = {s: actor_mesh(m,
                       color=np.array(name_unsided_to_color[convert_to_unsided_label(s)])/255.,
#                                      opacity=(0.2 if s == 'SNR_L' else 0.03),
                                           opacity=(0.95 if s == 'SNR_L' else 0.03),
                                    )
            for s, m in atlas_meshes_um.iteritems()}

shell_polydata_10um_wrt_canonicalAtlasSpace = DataManager.load_mesh_v2(brain_spec=dict(name=atlas_name, vol_type='score', resolution='10.0um'),
                                                                   structure='shell')

shell_polydata_um_wrt_canonicalAtlasSpace = rescale_polydata(shell_polydata_10um_wrt_canonicalAtlasSpace, 10.)

shell_actor_um_wrt_canonicalAtlasSpace = actor_mesh(shell_polydata_um_wrt_canonicalAtlasSpace, (1,1,1), opacity=.15,
                              wireframe=False)

shell_actor_10um_wrt_canonicalAtlasSpace = actor_mesh(shell_polydata_10um_wrt_canonicalAtlasSpace, (1,1,1), opacity=.15,
                              wireframe=False)

# launch_vtk(atlas_structure_actors_um.values() \
#            + [shell_actor_um_wrt_canonicalAtlasSpace] \
#            + [actor_sphere(position=(0,0,0), radius=5, color=(1,1,1), opacity=1.)])



marker_resolution = '10.0um'

markers_rel2atlas_actors = {}
aligned_markers_rel2atlas_um_all_brains = {}

for brain_name, experiment_info in experiments.iteritems():
    # Load Neurolucida format.

    markers = load_data(DataManager.get_lauren_markers_filepath(brain_name, structure='All', resolution=marker_resolution))

#     sample_n = min(len(markers), max(len(markers)/5, 10))
    sample_n = min(len(markers), 50)
    print brain_name, 'showing', sample_n, '/', len(markers)
    markers = markers[np.random.choice(range(len(markers)), size=sample_n, replace=False)]

    brain_f_spec = dict(name=brain_name, vol_type='annotationAsScore', structure='SNR_L', resolution='10.0um')
    brain_m_spec = dict(name=atlas_name, resolution='10.0um', vol_type='score', structure='SNR_L')
    alignment_spec = dict(stack_m=brain_m_spec, stack_f=brain_f_spec, warp_setting=7)

    tf_atlas_to_subj = DataManager.load_alignment_results_v3(alignment_spec, what='parameters', out_form=(4,4))

    markers_rel2subj = {marker_id: marker_xyz for marker_id, marker_xyz in enumerate(markers)}

    aligned_markers_rel2atlas = {marker_ind: transform_points(pts=p, transform=np.linalg.inv(tf_atlas_to_subj))
                                for marker_ind, p in markers_rel2subj.iteritems()}

    aligned_markers_rel2atlas_um = {marker_ind: p * convert_resolution_string_to_um(marker_resolution)
                                    for marker_ind, p in aligned_markers_rel2atlas.iteritems()}

    aligned_markers_rel2atlas_um_all_brains[brain_name] = aligned_markers_rel2atlas_um

    markers_rel2atlas_actors[brain_name] = [actor_sphere(position=(x,y,z), radius=20,
                                                        color=literal_eval(experiment_info['marker_color']),
                                                        opacity=.8 )
                               for marker_id, (x,y,z) in aligned_markers_rel2atlas_um.iteritems()]


launch_vtk([m for b, marker_actors in markers_rel2atlas_actors.iteritems() for m in marker_actors ] \
          + atlas_structure_actors_um.values() \
           + [shell_actor_um_wrt_canonicalAtlasSpace] \
#            + [actor_sphere(position=(0,0,0), radius=5, color=(1,1,1), opacity=1.)]
           ,
           init_angle='sagittal'
          )
