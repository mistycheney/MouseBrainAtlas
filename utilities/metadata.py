"""
This module stores static meta information.
"""
import os, sys
import numpy as np
import subprocess

########### Data Directories #############

hostname = subprocess.check_output("hostname", shell=True).strip()

# if hostname.endswith('sdsc.edu'):
#     print 'Setting environment for Gordon'
#     ROOT_DIR = '/oasis/projects/nsf/csd395/yuncong'
#     RAW_DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data'
#     data_dir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
#     DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
#     thumbnail_data_dir = data_dir
#     THUMBNAIL_DATA_DIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_processed'
#     REGISTRATION_PARAMETERS_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_registration_parameters'
#     REGISTRATION_VIZ_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_registration_visualization'
#     VOLUME_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_volumes'
#     labelingViz_root = '/oasis/projects/nsf/csd395/yuncong/CSHL_annotationsViz'
#     annotationViz_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_annotationsViz'
#     annotation_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_labelings_losslessAlignCropped/'
#     annotation_midbrainIncluded_v2_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_labelings_v3/'
#     ANNOTATION_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_labelings_v3/'
#     patch_training_features_rootdir = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2_train'
#     patch_rootdir = '/oasis/projects/nsf/csd395/yuncong/CSHL_data_patches/'
#     CLF_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_classifiers'
#     CLF_NISSL_ROOTDIR = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers'
#     CLF_NTBLUE_ROOTDIR = '/home/yuncong/csd395/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers_neurotraceBlue'
#     CELLS_ROOTDIR = '/home/yuncong/csd395/CSHL_cells_v2'
#     DETECTED_CELLS_ROOTDIR = os.path.join(CELLS_ROOTDIR, 'detected_cells')
#     CELL_EMBEDDING_ROOTDIR = os.path.join(CELLS_ROOTDIR, 'embedding')
#     D3JS_ROOTDIR = os.path.join(CELLS_ROOTDIR, 'd3js')
#     CELL_FEATURES_CLF_ROOTDIR = os.path.join(CELLS_ROOTDIR, 'classifiers')

#     PATCH_FEATURES_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_patch_features'
#     SPARSE_SCORES_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_sparse_scoremaps'
#     SCOREMAPS_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_dense_scoremaps'
#     SCOREMAP_VIZ_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_scoremap_viz'
#     HESSIAN_ROOTDIR = '/oasis/projects/nsf/csd395/yuncong/CSHL_hessians'
#     WORKSTATION_ROOTDIR = '/media/yuncong/BstemAtlasData/CSHL_data_processed'

#     S3_DATA_BUCKET = 'mousebrainatlas-data'
#     REPO_DIR = os.environ['REPO_DIR']
#     ON_AWS = False
#     ELASTIX_BIN = '/oasis/projects/nsf/csd395/yuncong/elastix_linux64_v4.7/bin/elastix'

if hostname == 'yuncong-MacbookPro':
    print 'Setting environment for Local Macbook Pro'
    HOST_ID = 'localhost'

    # REPO_DIR = '/home/yuncong/Brain' # use os.environ['REPO_DIR'] instead
    REPO_DIR = os.environ['REPO_DIR']
    ROOT_DIR = '/home/yuncong'
    DATA_ROOTDIR = '/media/yuncong/YuncongPublic/'
    THUMBNAIL_DATA_ROOTDIR = ROOT_DIR

    RAW_DATA_DIR = os.path.join(ROOT_DIR, 'CSHL_data')
    DATA_DIR = os.path.join(DATA_ROOTDIR, 'CSHL_data_processed')
    THUMBNAIL_DATA_DIR = os.path.join(THUMBNAIL_DATA_ROOTDIR, 'CSHL_data_processed')

    # VOLUME_ROOTDIR = '/home/yuncong/CSHL_volumes'
    VOLUME_ROOTDIR = os.path.join(DATA_ROOTDIR, 'CSHL_volumes')
    MESH_ROOTDIR =  '/home/yuncong/CSHL_meshes'
    REGISTRATION_PARAMETERS_ROOTDIR = '/home/yuncong/CSHL_registration_parameters'

    PATCH_FEATURES_ROOTDIR = os.path.join(DATA_ROOTDIR, 'CSHL_patch_features')

    ANNOTATION_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_labelings_v3')
    ANNOTATION_THALAMUS_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_labelings_thalamus')
    CLF_ROOTDIR =  os.path.join(ROOT_DIR, 'CSHL_classifiers')

    S3_DATA_BUCKET = 'mousebrainatlas-data'
    S3_DATA_DIR = 'CSHL_data_processed'
    S3_RAWDATA_BUCKET = 'mousebrainatlas-rawdata'

    CLASSIFIER_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'classifier_settings.csv')
    DATASET_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'dataset_settings.csv')
    REGISTRATION_SETTINGS_CSV = os.path.join(REPO_DIR, 'registration', 'registration_settings.csv')
    PREPROCESS_SETTINGS_CSV = os.path.join(REPO_DIR, 'preprocess', 'preprocess_settings.csv')
    DETECTOR_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'detector_settings.csv')

    SPARSE_SCORES_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_patch_scores')
    PATCH_LOCATIONS_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_patch_locations')
    SCOREMAP_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_scoremaps')
    SCOREMAP_VIZ_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_scoremap_viz')

    LABELED_NEURONS_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_labeled_neurons')

    ELASTIX_BIN = 'elastix'


elif hostname == 'yuncong-Precision-WorkStation-T7500':
    print 'Setting environment for Precision WorkStation'
    HOST_ID = 'workstation'

    if 'ROOT_DIR' in os.environ:
        ROOT_DIR = os.environ['ROOT_DIR']
    else:
        ROOT_DIR = '/home/yuncong/'

    if 'DATA_ROOTDIR' in os.environ:
        DATA_ROOTDIR = os.environ['DATA_ROOTDIR']
    else:
        DATA_ROOTDIR = '/media/yuncong/BstemAtlasData'

    if 'THUMBNAIL_DATA_ROOTDIR' in os.environ:
        THUMBNAIL_DATA_ROOTDIR = os.environ['THUMBNAIL_DATA_ROOTDIR']
    else:
        THUMBNAIL_DATA_ROOTDIR = '/media/yuncong/BstemAtlasData'

    RAW_DATA_DIR = os.path.join(DATA_ROOTDIR, 'CSHL_data')

    ON_AWS = False
    S3_DATA_BUCKET = 'mousebrainatlas-data'
    S3_RAWDATA_BUCKET = 'mousebrainatlas-rawdata'

    REPO_DIR = os.environ['REPO_DIR']

    DATA_DIR = os.path.join(DATA_ROOTDIR, 'CSHL_data_processed')

    THUMBNAIL_DATA_DIR = os.path.join(THUMBNAIL_DATA_ROOTDIR, 'CSHL_data_processed')
    VOLUME_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_volumes')
    MESH_ROOTDIR =  '/home/yuncong/CSHL_meshes'

    # annotation_rootdir =  os.path.join(ROOT_DIR, 'CSHL_data_labelings_losslessAlignCropped')
#     annotation_midbrainIncluded_v2_rootdir = '/home/yuncong/CSHL_labelings_v3/'
    PATCH_FEATURES_ROOTDIR = os.path.join(DATA_ROOTDIR, 'CSHL_patch_features')
    PATCH_LOCATIONS_ROOTDIR = os.path.join(DATA_ROOTDIR, 'CSHL_patch_locations')

    SCOREMAP_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_scoremaps')
    SCOREMAP_VIZ_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_scoremap_viz')
    SPARSE_SCORES_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_patch_scores')

    ANNOTATION_ROOTDIR =  os.path.join(ROOT_DIR, 'CSHL_labelings_v3')
    ANNOTATION_THALAMUS_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_labelings_thalamus')
    CLF_ROOTDIR =  os.path.join(DATA_ROOTDIR, 'CSHL_classifiers')

    REGISTRATION_PARAMETERS_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_registration_parameters')
    REGISTRATION_VIZ_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_registration_visualization')

    KDU_EXPAND_BIN = '/home/yuncong/KDU7A2_Demo_Apps_for_Centos7-x86-64_170827/kdu_expand'

    CLASSIFIER_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'classifier_settings.csv')
    DATASET_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'dataset_settings.csv')
    REGISTRATION_SETTINGS_CSV = os.path.join(REPO_DIR, 'registration', 'registration_settings.csv')
    PREPROCESS_SETTINGS_CSV = os.path.join(REPO_DIR, 'preprocess', 'preprocess_settings.csv')
    DETECTOR_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'detector_settings.csv')

    MXNET_MODEL_ROOTDIR = os.path.join(ROOT_DIR, 'mxnet_models')

    ELASTIX_BIN = 'elastix'


elif hostname.startswith('ip'):
    print 'Setting environment for AWS compute node'
    HOST_ID = 'ec2'

    if 'ROOT_DIR' in os.environ:
        ROOT_DIR = os.environ['ROOT_DIR']
    else:
        ROOT_DIR = '/shared'

    if 'DATA_ROOTDIR' in os.environ:
        DATA_ROOTDIR = os.environ['DATA_ROOTDIR']
    else:
        DATA_ROOTDIR = '/shared'

    if 'THUMBNAIL_DATA_ROOTDIR' in os.environ:
        THUMBNAIL_DATA_ROOTDIR = os.environ['THUMBNAIL_DATA_ROOTDIR']
    else:
        THUMBNAIL_DATA_ROOTDIR = '/shared'

    ON_AWS = True
    S3_DATA_BUCKET = 'mousebrainatlas-data'
    S3_RAWDATA_BUCKET = 'mousebrainatlas-rawdata'
    S3_DATA_DIR = 'CSHL_data_processed'
    REPO_DIR = os.environ['REPO_DIR']
    RAW_DATA_DIR = os.path.join(ROOT_DIR, 'CSHL_data')
    DATA_DIR = os.path.join(DATA_ROOTDIR, 'CSHL_data_processed')
    THUMBNAIL_DATA_DIR = os.path.join(THUMBNAIL_DATA_ROOTDIR, 'CSHL_data_processed')
    VOLUME_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_volumes')

    MESH_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_meshes')

    # SCOREMAP_VIZ_ROOTDIR = '/shared/CSHL_scoremap_viz_Sat16ClassFinetuned_v2'
    ANNOTATION_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_labelings_v3')
    ANNOTATION_THALAMUS_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_labelings_thalamus')
    ANNOTATION_VIZ_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_annotation_viz')
    # SVM_ROOTDIR = '/shared/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers/'
    # SVM_NTBLUE_ROOTDIR = '/shared/CSHL_patch_features_Sat16ClassFinetuned_v2_classifiers_neurotraceBlue/'
    PATCH_FEATURES_ROOTDIR = os.path.join(DATA_ROOTDIR, 'CSHL_patch_features')
    PATCH_LOCATIONS_ROOTDIR = os.path.join(DATA_ROOTDIR, 'CSHL_patch_locations')
    SCOREMAP_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_scoremaps')
    SCOREMAP_VIZ_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_scoremap_viz')
    SPARSE_SCORES_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_patch_scores')
    REGISTRATION_PARAMETERS_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_registration_parameters')
    REGISTRATION_VIZ_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_registration_visualization')
    # SPARSE_SCORES_ROOTDIR = '/shared/CSHL_patch_Sat16ClassFinetuned_v2_predictions'
    # SCOREMAPS_ROOTDIR = '/shared/CSHL_lossless_scoremaps_Sat16ClassFinetuned_v2'
    # HESSIAN_ROOTDIR = '/shared/CSHL_hessians/'
    ELASTIX_BIN = 'elastix'
    KDU_EXPAND_BIN = '/home/ubuntu/KDU79_Demo_Apps_for_Linux-x86-64_170108/kdu_expand'
    CELLPROFILER_EXEC = 'python /shared/CellProfiler/CellProfiler.py' # /usr/local/bin/cellprofiler
    CELLPROFILER_PIPELINE_FP = '/shared/CSHL_cells_v2/SegmentCells.cppipe'

    if 'CELLS_ROOTDIR' in os.environ:
        CELLS_ROOTDIR = os.environ['CELLS_ROOTDIR']
    else:
        CELLS_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_cells_v2')

    DETECTED_CELLS_ROOTDIR = os.path.join(CELLS_ROOTDIR, 'detected_cells')
    CELL_EMBEDDING_ROOTDIR = os.path.join(CELLS_ROOTDIR, 'embedding')
    D3JS_ROOTDIR = os.path.join(CELLS_ROOTDIR, 'd3js')
    CELL_FEATURES_CLF_ROOTDIR = os.path.join(CELLS_ROOTDIR, 'classifiers')

    CLF_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_classifiers')

    CLASSIFIER_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'classifier_settings.csv')
    DATASET_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'dataset_settings.csv')
    REGISTRATION_SETTINGS_CSV = os.path.join(REPO_DIR, 'registration', 'registration_settings.csv')
    PREPROCESS_SETTINGS_CSV = os.path.join(REPO_DIR, 'preprocess', 'preprocess_settings.csv')
    DETECTOR_SETTINGS_CSV = os.path.join(REPO_DIR, 'learning', 'detector_settings.csv')

    MXNET_MODEL_ROOTDIR = os.path.join(ROOT_DIR, 'mxnet_models')

    LABELED_NEURONS_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_labeled_neurons')

    CSHL_SPM_ROOTDIR = os.path.join(ROOT_DIR, 'CSHL_SPM')

else:
    print 'Setting environment for Brainstem workstation'


#################### Resolution conversions ############

wrt_metadata = {'wholebrain': {'origin_wrt_wholebrain_um': 0, 
                'plane': 'sagittal', 
                'zdim_um': None}}

def convert_frame(p, in_frame, out_frame, zdim):
    """
    Convert among the three frames specified by the second methods here
    https://docs.google.com/presentation/d/1o5aQbXY5wYC0BNNiEZm7qmjvngbD_dVoMyCw_tAQrkQ/edit#slide=id.g2d31ede24d_0_0
    """

    if in_frame == 'sagittal':
        p_sagittal = p
    elif in_frame == 'coronal':
        x = p[..., 2]
        y = p[..., 1]
        z = zdim - p[..., 0]
        p_sagittal = np.column_stack([x,y,z])
    elif in_frame == 'horizontal':
        x = p[..., 0]
        y = p[..., 2]
        z = zdim - p[..., 1]
        p_sagittal = np.column_stack([x,y,z])
    else:
        print in_frame
        raise

    if out_frame == 'sagittal':
        p_out = p_sagittal
    elif out_frame == 'coronal':
        x = zdim - p_sagittal[..., 2]
        y = p_sagittal[..., 1]
        z = p_sagittal[..., 0]
        p_out = np.column_stack([x,y,z])
    elif out_frame == 'horizontal':
        x = p_sagittal[..., 0]
        y = zdim - p_sagittal[..., 2]
        z = p_sagittal[..., 1]
        p_out = np.column_stack([x,y,z])
    else:
        print out_frame
        raise

    return p_out

def convert_resolution(p, in_resolution, out_resolution, 
                       stack=None, image_resolution=None, 
                       section_list=None,
                      volume_resolution_um=None):

    if in_resolution == 'image':
        p_um = p * convert_resolution_string_to_um(stack=stack, resolution=image_resolution)
    elif in_resolution == 'image_image_section':
        uv_um = p[..., :2] * convert_resolution_string_to_um(stack=stack, resolution=image_resolution)
        d_um = np.array([SECTION_THICKNESS * sec for sec in p[..., 2]])
        p_um = np.column_stack([uv_um, d_um])
    elif in_resolution == 'image_image_index':
        uv_um = p[..., :2] * convert_resolution_string_to_um(stack=stack, resolution=image_resolution)
        i_um = np.array([SECTION_THICKNESS * section_list[int(idx)] for idx in p[..., 2]])
        p_um = np.column_stack([uv_um, i_um])
    elif in_resolution == 'volume':
        p_um = p * volume_resolution_um
    elif in_resolution == 'raw':
        p_um = p * planar_resolution[stack]
    elif in_resolution == 'down32':
        p_um = p * (planar_resolution[stack] * 32.)
    elif in_resolution == 'um':
        p_um = p
    else:
        raise

    if out_resolution == 'image':
        p_outResol = p_um / convert_resolution_string_to_um(stack=stack, resolution=image_resolution)
    # elif out_resolution == 'image_image_section':
    #     uv_outResol = p_um[..., :2] / convert_resolution_string_to_voxel_size(stack=self.gui.stack, resolution=self.data_feeder.resolution)
    #     sec_outResol = np.array([1 + int(np.floor(d_um / SECTION_THICKNESS)) for d_um in p_um[..., 2]])
    #     p_outResol = np.column_stack([uv_outResol, sec_outResol])
    elif out_resolution == 'image_image_index':
        uv_outResol = p_um[..., :2] / convert_resolution_string_to_um(stack=stack, resolution=image_resolution)
        if hasattr(self.data_feeder, 'sections'):
            i_outResol = []
            for d_um in p_um[..., 2]:
                sec = 1 + int(np.floor(d_um / SECTION_THICKNESS))
                if sec in section_list:
                    index = section_list.index(sec)
                else:
                    index = np.nan
                i_outResol.append(index)
            i_outResol = np.array(i_outResol)
            # i_outResol = np.array([self.data_feeder.sections.index(1 + int(np.floor(d_um / SECTION_THICKNESS))) for d_um in p_um[..., 2]])
        else:
            i_outResol = p_um[..., 2] / convert_resolution_string_to_um(stack=stack, resolution=image_resolution)
        p_outResol = np.column_stack([uv_outResol, i_outResol])
    elif out_resolution == 'volume':
        p_outResol = p_um / volume_resolution_um
    elif out_resolution == 'raw':
        p_outResol = p_um / planar_resolution[stack]
    elif out_resolution == 'down32':
        p_outResol = p_um / (planar_resolution[stack] * 32.)
    elif out_resolution == 'um':
        p_outResol = p_um
    else:
        raise

    return p_outResol


def convert_from_wholebrain_um(p_wrt_wholebrain_um, wrt, resolution,
                               image_resolution=None, stack=None, volume_resolution_um=None,
    structure_origin=None, structure_wrt=None, structure_resolution=None, structure_zdim=None):

    p_wrt_wholebrain_um = np.array(p_wrt_wholebrain_um)
    assert np.atleast_2d(p_wrt_wholebrain_um).shape[1] == 3

    if wrt == 'wholebrain':
        p_wrt_outdomain_um = p_wrt_wholebrain_um
    elif 'sagittal' in wrt or 'coronal' in wrt or 'horizontal' in wrt:
        box, plane = wrt.split('_')
        p_wrt_boxSagittal_origin_um = p_wrt_wholebrain_um - wrt_metadata[wrt]['origin_wrt_wholebrain_um']
        assert plane == 'sagittal', plane # otherwise, need to provide zdim to convert_frame.
        p_wrt_outdomain_um = convert_frame(p_wrt_boxSagittal_origin_um, in_frame='sagittal', 
                                           out_frame=plane, 
                                           zdim=wrt_metadata[wrt]['zdim_um'])
    else:
        print wrt
        raise

    p_wrt_outdomain_outResol = convert_resolution(p_wrt_outdomain_um, in_resolution='um', out_resolution=resolution,
                                                 image_resolution=image_resolution, stack=stack, volume_resolution_um=volume_resolution_um)
    return np.squeeze(p_wrt_outdomain_outResol)

def convert_to_wholebrain_um(p, wrt, resolution,
                             image_resolution=None, stack=None, volume_resolution_um=None,
    structure_origin=None, structure_wrt=None, structure_resolution=None, structure_zdim=None):

    p = np.array(p)
    assert np.atleast_2d(p).shape[1] == 3

    p_um = convert_resolution(p, in_resolution=resolution, out_resolution='um',
                             image_resolution=image_resolution, stack=stack, volume_resolution_um=volume_resolution_um)

    if wrt == 'wholebrain':
        p_wrt_wholebrain_um = p_um
    elif 'sagittal' in wrt or 'coronal' in wrt or 'horizontal' in wrt:
        box, plane = wrt.split('_')
        assert plane == 'sagittal', plane # otherwise, need to provide zdim to convert_frame.
        p_wrt_boxSagittal_um = convert_frame(p_um, in_frame=plane, out_frame='sagittal', zdim=wrt_metadata[wrt]['zdim_um'])
        box_origin_wrt_wholebrain_um = wrt_metadata[wrt]['origin_wrt_wholebrain_um']
        p_wrt_wholebrain_um = p_wrt_boxSagittal_um + box_origin_wrt_wholebrain_um
    else:
        print wrt
        raise

    return np.squeeze(p_wrt_wholebrain_um)

def convert_frame_and_resolution(p, in_wrt, in_resolution, out_wrt, out_resolution,
                                 image_resolution=None, stack=None, volume_resolution_um=None,
    structure_origin=None, structure_wrt=None, structure_resolution=None, structure_zdim=None):
        """
        `wrt` can be any of:
        - wholebrain
        - sagittal: frame of lo-res sagittal scene = sagittal frame of the intensity volume, with origin at the most left/rostral/dorsal position.
        - coronal: frame of lo-res coronal scene = coronal frame of the intensity volume, with origin at the most left/rostral/dorsal position.
        - horizontal: frame of lo-res horizontal scene = horizontal frame of the intensity volume, with origin at the most left/rostral/dorsal position.

        `resolution` can be any of:
        - raw
        - down32
        - vol
        - image: gscene resolution, determined by data_feeder.resolution
        - image_image_index: (u in image resolution, v in image resolution, i in terms of data_feeder index)
        """

        p_wrt_wholebrain_um = convert_to_wholebrain_um(p, wrt=in_wrt, resolution=in_resolution,
                                                       image_resolution=image_resolution, stack=stack, volume_resolution_um=volume_resolution_um,
        structure_origin=structure_origin, structure_wrt=structure_wrt, structure_resolution=structure_resolution, structure_zdim=structure_zdim)

        p_wrt_outdomain_outResol = convert_from_wholebrain_um(p_wrt_wholebrain_um=p_wrt_wholebrain_um, wrt=out_wrt, resolution=out_resolution,
                                                              image_resolution=image_resolution, stack=stack, volume_resolution_um=volume_resolution_um,
        structure_origin=structure_origin, structure_wrt=structure_wrt, structure_resolution=structure_resolution, structure_zdim=structure_zdim)
        # print 'p', p
        # print "p_wrt_wholebrain_um", p_wrt_wholebrain_um
        # print 'p_wrt_outdomain_outResol', p_wrt_outdomain_outResol
        return p_wrt_outdomain_outResol

def convert_resolution_string_to_um(resolution, stack=None):
    return convert_resolution_string_to_voxel_size(resolution, stack=stack)

def convert_resolution_string_to_voxel_size(resolution, stack=None):
    """
    Args:
        resolution (str):
    Returns:
        voxel/pixel size in microns.
    """
    if resolution in ['down32', 'thumbnail']:
        assert stack is not None
        return planar_resolution[stack] * 32.        
    elif resolution == 'lossless' or resolution == 'down1' or resolution == 'raw':
        assert stack is not None
        return planar_resolution[stack] 
    elif resolution == 'down8':
        assert stack is not None
        return planar_resolution[stack] * 8.
    elif resolution.endswith('um'):
        return float(resolution[:-2])
    else:
        print resolution
        raise Exception("Unknown resolution string %s" % resolution)

#################### Name conversions ##################

def parse_label(label, singular_as_s=False):
    """
    Args:
        singular_as_s (bool): If true, singular structures have side = 'S', otherwise side = None.

    Returns:
        (structure name, side, surround margin, surround structure name)
    """
    import re
    try:
        m = re.match("([0-9a-zA-Z]*)(_(L|R))?(_surround_([0-9]+))?(_([0-9a-zA-Z]*))?", label)
    except:
        raise Exception("Parse label error: %s" % label)
    g = m.groups()
    structure_name = g[0]
    side = g[2]
    if side is None:
        if singular_as_s:
            side = 'S'
    surround_margin = g[4]
    surround_structure_name = g[6]
    return structure_name, side, surround_margin, surround_structure_name

is_sided_label = lambda label: parse_label(label)[1] is not None
is_surround_label = lambda label: parse_label(label)[2] is not None
get_side_from_label = lambda label: parse_label(label)[1]
get_margin_from_label = lambda label: parse_label(label)[2]

def compose_label(structure_name, side=None, surround_margin=None, surround_structure_name=None):
    label = structure_name
    if side is not None:
        label += '_' + side
    if surround_margin is not None:
        label += '_surround_' + surround_margin
    if surround_structure_name is not None:
        label += '_' + surround_structure_name
    return label

def convert_to_unsided_label(label):
    structure_name, side, surround_margin, surround_structure_name = parse_label(label)
    return compose_label(structure_name, side=None, surround_margin=surround_margin, surround_structure_name=surround_structure_name)

def convert_to_nonsurround_label(name):
    return convert_to_nonsurround_name(name)

    # return convert_name_to_unsided(name)

# def convert_name_to_unsided(name):
#     if '_' not in name:
#         return name
#     else:
#         return convert_to_original_name(name)

def convert_to_left_name(name):
    if name in singular_structures:
        sys.stderr.write("Asked for left name for singular structure %s, returning itself.\n" % name)
        return name
    else:
        return convert_to_unsided_label(name) + '_L'

def convert_to_right_name(name):
    if name in singular_structures:
        sys.stderr.write("Asked for right name for singular structure %s, returning itself.\n" % name)
        return name
    else:
        return convert_to_unsided_label(name) + '_R'

def convert_to_original_name(name):
    return name.split('_')[0]

def convert_to_nonsurround_name(name):
    if is_surround_label(name):
        import re
        m = re.match('(.*?)_surround_.*', name)
        return m.groups()[0]
    else:
        return name

def convert_to_surround_name(name, margin=None, suffix=None):

    elements = name.split('_')
    if margin is None:
        if len(elements) > 1 and elements[1] == 'surround':
            if suffix is not None:
                return elements[0] + '_surround_' + suffix
            else:
                return elements[0] + '_surround'
        else:
            if suffix is not None:
                return name + '_surround_' + suffix
            else:
                return name + '_surround'
    else:
        if len(elements) > 1 and elements[1] == 'surround':
            if suffix is not None:
                return elements[0] + '_surround_' + str(margin) + '_' + suffix
            else:
                return elements[0] + '_surround_' + str(margin)
        else:
            if suffix is not None:
                return name + '_surround_' + str(margin) + '_' + suffix
            else:
                return name + '_surround_' + str(margin)


#######################################

from pandas import read_csv
dataset_settings = read_csv(DATASET_SETTINGS_CSV, header=0, index_col=0)
classifier_settings = read_csv(CLASSIFIER_SETTINGS_CSV, header=0, index_col=0)
registration_settings = read_csv(REGISTRATION_SETTINGS_CSV, header=0, index_col=0)
preprocess_settings = read_csv(PREPROCESS_SETTINGS_CSV, header=0, index_col=0)
detector_settings = read_csv(DETECTOR_SETTINGS_CSV, header=0, index_col=0)
windowing_settings = {1: {"patch_size": 224, "spacing": 56},
                      2: {'patch_size':224, 'spacing':56, 'comment':'larger margin'},
                     3: {'patch_size':224, 'spacing':32, 'comment':'smaller spacing'},
                     4: {'patch_size':224, 'spacing':128, 'comment':'smaller spacing'},
                     5: {'patch_size':224, 'spacing':64, 'comment':'smaller spacing'},
                     6: {'patch_size': 448, 'spacing':64, 'comment': 'twice as large patch'},
                     7: {'patch_size_um':103.04, 'spacing_um':30, 'comment':'specify size/spacing in terms of microns rather than pixels'},
                     8: {'patch_size_um':206.08, 'spacing_um':30, 'comment':'larger patch'},
                     9: {'patch_size_um':412.16, 'spacing_um':30, 'comment':'larger patch'},
                    10: {'patch_size_um':824.32, 'spacing_um':30, 'comment':'larger patch'}
                     }

############ Class Labels #############

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', 'SNC', 'SNR', '3N', '4N',
                    'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
# singular_structures = ['AP', '12N', 'RtTg', 'sp5', 'outerContour', 'SC', 'IC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']
all_known_structures = paired_structures + singular_structures
all_known_structures_sided = sum([[n] if n in singular_structures
                        else [convert_to_left_name(n), convert_to_right_name(n)]
                        for n in all_known_structures], [])
#all_known_structures_sided_surround_only = [convert_to_surround_name(s, margin='x1.5') for s in all_known_structures_sided]
all_known_structures_sided_surround_only = [convert_to_surround_name(s, margin=200) for s in all_known_structures_sided]
all_known_structures_sided_with_surround = sorted(all_known_structures_sided + all_known_structures_sided_surround_only)
all_structures_with_classifiers = sorted([l for l in all_known_structures if l not in {'outerContour', 'sp5'}])

structures_sided_sorted_by_size = ['4N_L', '4N_R', '6N_L', '6N_R', 'Amb_L', 'Amb_R', 'PBG_L', 'PBG_R', '10N_L', '10N_R', 'AP', '3N_L', '3N_R', 'LC_L', 'LC_R', 'SNC_L', 'SNC_R', 'Tz_L', 'Tz_R', '7n_L', '7n_R', 'RMC_L', 'RMC_R', '5N_L', '5N_R', 'VCP_L', 'VCP_R', '12N', 'LRt_L', 'LRt_R', '7N_L', '7N_R', 'VCA_L', 'VCA_R', 'VLL_L', 'VLL_R', 'DC_L', 'DC_R', 'Sp5O_L', 'Sp5O_R', 'Sp5I_L', 'Sp5I_R', 'Pn_L', 'Pn_R', 'RtTg', 'SNR_L', 'SNR_R', 'Sp5C_L', 'Sp5C_R', 'IC', 'SC']

structures_sided_sorted_by_rostral_caudal_position = ['SNC_R', 'SNC_L', 'SC', 'SNR_R', 'SNR_L', 'RMC_R', 'RMC_L', '3N_R', '3N_L', 'PBG_R', 'PBG_L', '4N_R', '4N_L', 'Pn_R', 'Pn_L', 'VLL_R', 'VLL_L', 'RtTg', '5N_R', '5N_L', 'LC_R', 'LC_L', 'Tz_R', 'Tz_L', 'VCA_R', 'VCA_L', '7n_R', '7n_L', '6N_R', '6N_L', 'DC_R', 'VCP_R', 'DC_L', 'VCP_L', '7N_R', '7N_L', 'Sp5O_R', 'Sp5O_L', 'Amb_R', 'Amb_L', 'Sp5I_R', 'Sp5I_L', 'AP', '12N', '10N_R', '10N_L', 'LRt_R', 'LRt_L', 'Sp5C_R', 'Sp5C_L']

structures_unsided_sorted_by_rostral_caudal_position = ['SNC', 'SC', 'IC', 'SNR', 'RMC', '3N', 'PBG','4N', 'Pn','VLL','RtTg', '5N', 'LC', 'Tz', 'VCA', '7n', '6N', 'DC', 'VCP', '7N', 'Sp5O', 'Amb', 'Sp5I', 'AP', '12N', '10N', 'LRt', 'Sp5C']

#linear_landmark_names_unsided = ['outerContour']
linear_landmark_names_unsided = []
volumetric_landmark_names_unsided = list(set(paired_structures + singular_structures) - set(linear_landmark_names_unsided))
all_landmark_names_unsided = volumetric_landmark_names_unsided + linear_landmark_names_unsided

labels_unsided = volumetric_landmark_names_unsided + linear_landmark_names_unsided
labels_unsided_indices = dict((j, i+1) for i, j in enumerate(labels_unsided))  # BackG always 0

labelMap_unsidedToSided = dict([(name, [name+'_L', name+'_R']) for name in paired_structures] + \
                            [(name, [name]) for name in singular_structures])

labelMap_sidedToUnsided = {n: nu for nu, ns in labelMap_unsidedToSided.iteritems() for n in ns}

from itertools import chain
labels_sided = list(chain(*(labelMap_unsidedToSided[name_u] for name_u in labels_unsided)))
labels_sided_indices = dict((j, i+1) for i, j in enumerate(labels_sided)) # BackG always 0

############ Physical Dimension #############

# section_thickness = 20 # in um
SECTION_THICKNESS = 20. # in um
# xy_pixel_distance_lossless = 0.46
XY_PIXEL_DISTANCE_LOSSLESS = 0.46 # This is the spec for Nanozoomer
XY_PIXEL_DISTANCE_TB = XY_PIXEL_DISTANCE_LOSSLESS * 32 # in um, thumbnail

# This is the spec for Axioscan (our data)
XY_PIXEL_DISTANCE_LOSSLESS_AXIOSCAN = 0.325 # unit is micron
XY_PIXEL_DISTANCE_TB_AXIOSCAN = XY_PIXEL_DISTANCE_LOSSLESS_AXIOSCAN * 32

#######################################

all_nissl_stacks = ['MD585', 'MD589', 'MD590', 'MD591', 'MD592', 'MD593', 'MD594', 'MD595', 'MD598', 'MD599', 'MD602', 'MD603']
all_ntb_stacks = ['MD635']
all_dk_ntb_stacks = ['ChatCryoJane201710']
all_alt_nissl_ntb_stacks = ['MD653', 'MD652', 'MD642']
all_alt_nissl_tracing_stacks = ['MD657', 'MD658', 'MD661', 'MD662']
# all_stacks = all_nissl_stacks + all_ntb_stacks
all_stacks = all_nissl_stacks + all_ntb_stacks + all_alt_nissl_ntb_stacks + all_alt_nissl_tracing_stacks + all_dk_ntb_stacks
all_annotated_nissl_stacks = ['MD585', 'MD589', 'MD594']
all_annotated_ntb_stacks = ['MD635']
all_annotated_stacks = all_annotated_nissl_stacks + all_annotated_ntb_stacks

planar_resolution = {'MD585': XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD589':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD590':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD591':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD592':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD593':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD594':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD595':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD598':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD599':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD602':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD603':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD635':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD653':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD652':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD642':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD657':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD658':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD661':XY_PIXEL_DISTANCE_LOSSLESS,
                     'MD662':XY_PIXEL_DISTANCE_LOSSLESS,
                     'ChatCryoJane201710': XY_PIXEL_DISTANCE_LOSSLESS_AXIOSCAN,
                     'DmaleAxioscan': XY_PIXEL_DISTANCE_LOSSLESS_AXIOSCAN,
                     'CHATM2': XY_PIXEL_DISTANCE_LOSSLESS_AXIOSCAN
                    }

#######################################

import multiprocessing
NUM_CORES = multiprocessing.cpu_count()

############## Colors ##############

from utilities2015 import high_contrast_colors
hc_perm = [ 0,  5, 28, 26, 12, 11,  4,  8, 25, 22,  3,  1, 20, 19, 27, 13, 24,
       17, 16, 15,  7, 14, 21, 18, 23,  2, 10,  9,  6]
high_contrast_colors = [high_contrast_colors[i] for i in hc_perm]
name_sided_to_color = {s: high_contrast_colors[i%len(high_contrast_colors)]
                     for i, s in enumerate(all_known_structures_sided) }
name_unsided_to_color = {s: high_contrast_colors[i%len(high_contrast_colors)]
                     for i, s in enumerate(all_known_structures) }
stack_to_color = {n: high_contrast_colors[i%len(high_contrast_colors)] for i, n in enumerate(all_stacks)}
