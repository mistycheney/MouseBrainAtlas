import sys, os
import subprocess
import os

try:
    import boto3
except:
    sys.stderr.write('No boto3\n')
from pandas import read_hdf

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from vis3d_utilities import *
from distributed_utilities import *

def is_invalid(fn=None, sec=None, stack=None):
    if sec is not None:
        assert stack is not None
        fn = metadata_cache['sections_to_filenames'][stack][sec]
    else:
        assert fn is not None
    return fn in ['Nonexisting', 'Rescan', 'Placeholder']

def volume_type_to_str(t):
    if t == 'score':
        return 'scoreVolume'
    elif t == 'annotation':
        return 'annotationVolume'
    elif t == 'annotation_as_score':
        return 'annotationAsScoreVolume'
    elif t == 'outer_contour':
        return 'outerContourVolume'
    elif t == 'intensity':
        return 'intensityVolume'
    else:
        raise Exception('Volume type %s is not recognized.' % t)

def generate_suffix(train_sample_scheme=None, global_transform_scheme=None, local_transform_scheme=None):

    suffix = []
    if train_sample_scheme is not None:
        suffix.append('trainSampleScheme_%d'%train_sample_scheme)
    if global_transform_scheme is not None:
        suffix.append('globalTxScheme_%d'%global_transform_scheme)
    if local_transform_scheme is not None:
        suffix.append('localTxScheme_%d'%local_transform_scheme)

    return '_'.join(suffix)

def save_file_to_s3(local_path, s3_path):
    # upload to s3
    return

def save_to_s3(fpkw, fppos):
    """
    Decorator. Must provide both `fpkw` and `fppos` because we don't know if
    filepath will be supplied to the decorated function as positional argument
    or keyword argument.

    fpkw: argument keyword for file path in the decorated function
    fppos: argument position for file path in the decorated function

    Reference: http://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html
    """
    def wrapper(func):
        def wrapped_f(*args, **kwargs):
            if fpkw in kwargs:
                fp = kwargs[fpkw]
            elif len(args) > fppos:
                fp = args[fppos]
            res = func(*args, **kwargs)
            save_file_to_s3(fp, DataManager.map_local_filename_to_s3(fp))
            return res
        return wrapped_f
    return wrapper

class DataManager(object):

    # @staticmethod
    # def map_local_filename_to_s3(local_fp):
    #     s3_path = local_fp.replace(os.path.dirname(data_dir), "s3://" + S3_DATA_BUCKET + '/' + )
    #     return s3_path

    @staticmethod
    def load_data(filepath, filetype):

        if not os.path.exists(filepath):
            sys.stderr.write('File does not exist: %s\n' % filepath)

            # If on aws, download from S3 and make available locally.
            # if ON_AWS:
                # DataManager.download_from_s3(filepath, DataManager.map_local_filename_to_s3(filepath))

        if filetype == 'bp':
            return bp.unpack_ndarray_file(filepath)
        elif filetype == 'image':
            return imread(filepath)
        elif filetype == 'hdf':
            return load_hdf(filepath)
        elif filetype == 'bbox':
            return np.loadtxt(filepath).astype(np.int)
        elif filetype == 'annotation_hdf':
            contour_df = read_hdf(filepath, 'contours')
            return contour_df
        elif filetype == 'pickle':
            import cPickle as pickle
            return pickle.load(open(filepath, 'r'))
        elif filetype == 'file_section_map':
            with open(filepath, 'r') as f:
                fn_idx_tuples = [line.strip().split() for line in f.readlines()]
                filename_to_section = {fn: int(idx) for fn, idx in fn_idx_tuples}
                section_to_filename = {int(idx): fn for fn, idx in fn_idx_tuples}
            return filename_to_section, section_to_filename
        elif filetype == 'label_name_map':
            label_to_name = {}
            name_to_label = {}
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    name_s, label = line.split()
                    label_to_name[int(label)] = name_s
                    name_to_label[name_s] = int(label)
            return label_to_name, name_to_label
        elif filetype == 'anchor':
            with open(filepath, 'r') as f:
                anchor_fn = f.readline()
            return anchor_fn
        elif filetype == 'transform_params':
            with open(filepath, 'r') as f:
                lines = f.readlines()

                global_params = one_liner_to_arr(lines[0], float)
                centroid_m = one_liner_to_arr(lines[1], float)
                xdim_m, ydim_m, zdim_m  = one_liner_to_arr(lines[2], int)
                centroid_f = one_liner_to_arr(lines[3], float)
                xdim_f, ydim_f, zdim_f  = one_liner_to_arr(lines[4], int)

            return global_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f
        else:
            sys.stderr.write('File type %s not recognized.\n' % filetype)

    # @staticmethod
    # def load_volume_bbox(stack):
    #     with open(os.path.join(volume_dir, stack, stack+'_down32_annotationVolume_bbox.txt'), 'r') as f:
    #         bbox = map(int, f.readline().strip().split())
    #     return bbox

    @staticmethod
    def get_anchor_filename_filename(stack):
        fn = THUMBNAIL_DATA_DIR + '/%(stack)s/%(stack)s_anchor.txt' % dict(stack=stack)
        return fn

    # @staticmethod
    # def download_from_s3(local_path, s3_path = None):
    # #downloading 500 files of 1Mb each
    # #boto3 - 36 seconds
    # #aws cli - 5 seconds
    #     s3_connection = boto3.resource('s3')
    #     if s3_path == None:
    #         s3_path = DataManager.map_local_filename_to_s3(local_path)
    #     bucket, file_to_download= s3_path.split("s3://")[1].split("/", 1)
    #     #file_to_download = file_to_download.split("/", 1)[1]
    #
    #     bucket = s3_connection.Bucket(bucket)
    #     create_parent_dir_if_not_exists(local_path)
    #     if len(list(bucket.objects.filter(Prefix=file_to_download))) > 1:
    #         execute_command('aws s3 cp --recursive %s %s' % (s3_path, local_path))
    #         #subprocess.call(["aws", "s3", "cp", s3_path, local_path, "--recursive"], stdout = open(os.devnull, 'w'))
    #     else:
    #         bucket.download_file(file_to_download, local_path)
    # return local_path

    # @staticmethod
    # def upload_to_s3(local_path, s3_path = None, output = False):
    # #uploading 500 files of 1Mb each
    # #boto3 - 1 minute 24  seconds
    # #aws cli - 7 seconds
    #     if s3_path == None:
    #         s3_path = map_local_filename_to_s3(local_path)
    #     if output == True:
    #         subprocess.call(["aws", "s3", "cp", local_path, s3_path, "--recursive"])
    #     else:
    #         subprocess.call(["aws", "s3", "cp", local_path, s3_path, "--recursive"], stdout = open(os.devnull, 'w'))

    @staticmethod
    def load_anchor_filename(stack):
        fn = DataManager.get_anchor_filename_filename(stack)
        anchor_fn = DataManager.load_data(fn, filetype='anchor')
        return anchor_fn

    @staticmethod
    def get_cropbox_filename(stack):
        fn = THUMBNAIL_DATA_DIR + '/%(stack)s/%(stack)s_cropbox.txt' % dict(stack=stack)
        return fn

    @staticmethod
    def load_cropbox(stack):
        fn = DataManager.get_cropbox_filename(stack=stack)
        cropbox = DataManager.load_data(fn, filetype='bbox')
        # with open(fn, 'r') as f:
        #     cropbox = one_liner_to_arr(f.readline(), int)
        return cropbox

    @staticmethod
    def get_sorted_filenames_filename(stack):
        fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_sorted_filenames.txt')
        return fn

    @staticmethod
    def load_sorted_filenames(stack):
        fn = DataManager.get_sorted_filenames_filename(stack)
        filename_to_section, section_to_filename = DataManager.load_data(fn, filetype='file_section_map')
        if 'Placeholder' in filename_to_section:
            filename_to_section.pop('Placeholder')
        return filename_to_section, section_to_filename

    @staticmethod
    def get_transforms_filename(stack):
        fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_transformsTo_anchor.pkl')
        return fn

    @staticmethod
    def load_transforms(stack, downsample_factor):
        """
        Load the transforms that when multiplied to a point on original space converts it to on aligned space.
        """

        fn = DataManager.get_transforms_filename(stack)
        Ts = DataManager.load_data(fn, filetype='pickle')

        Ts_inv_downsampled = {}
        for fn, T0 in Ts.iteritems():
            T = T0.copy()
            T[:2, 2] = T[:2, 2] * 32 / downsample_factor
            Tinv = np.linalg.inv(T)
            Ts_inv_downsampled[fn] = Tinv

        return Ts_inv_downsampled

    @staticmethod
    def get_thumbnail_mask_filename_v3(stack, section=None, version='aligned_cropped'):
        fp = DataManager.get_mask_filepath(stack=stack, sec=section, version=version)
        return fp

    @staticmethod
    def load_thumbnail_mask_v3(stack, section=None, version='aligned_cropped'):
        fn = DataManager.get_thumbnail_mask_filename_v3(stack=stack, section=section, version=version)
        mask = DataManager.load_data(fn, filetype='image').astype(np.bool)
        return mask
    
    @staticmethod
    def get_thumbnail_mask_filename_v2(stack, section=None, version='aligned_cropped'):        

        # anchor_fn = DataManager.load_anchor_filename(stack)
        anchor_fn = metadata_cache['anchor_fn'][stack]
        filename_to_section, section_to_filename = DataManager.load_sorted_filenames(stack)
        sections_to_filenames = metadata_cache['sections_to_filenames'][stack]
        fn = sections_to_filenames[section]
        
        if version == 'aligned_cropped':
            fn = THUMBNAIL_DATA_DIR+'/%(stack)s/%(stack)s_masks_alignedTo_%(anchor_fn)s_cropped/%(fn)s_mask_alignedTo_%(anchor_fn)s_cropped.png' % \
                dict(stack=stack, fn=fn, anchor_fn=anchor_fn)
        elif version == 'aligned':
            fn = THUMBNAIL_DATA_DIR+'/%(stack)s/%(stack)s_masks_alignedTo_%(anchor_fn)s/%(stack)s_%(sec)04d_mask_alignedTo_%(anchor_fn)s.png' % \
                dict(stack=stack, fn=fn, anchor_fn=anchor_fn)
        return fn

    @staticmethod
    def load_thumbnail_mask_v2(stack, section=None, version='aligned_cropped'):
        fn = DataManager.get_thumbnail_mask_filename_v2(stack=stack, section=section, version=version)
        mask = DataManager.load_data(fn, filetype='image').astype(np.bool)
        return mask

    @staticmethod
    def get_thumbnail_mask_filepath(stack, section, cerebellum_removed=False):
        if cerebellum_removed:
            fn = data_dir+'/%(stack)s_thumbnail_aligned_mask_cropped_cerebellumRemoved/%(stack)s_%(sec)04d_thumbnail_aligned_mask_cropped_cerebellumRemoved.png' % \
                {'stack': stack, 'sec': section}
        else:
            fn = data_dir+'/%(stack)s_thumbnail_aligned_mask_cropped/%(stack)s_%(sec)04d_thumbnail_aligned_mask_cropped.png' % \
                            {'stack': stack, 'sec': section}
        return fn

    @staticmethod
    def get_original_volume_basename(stack, classifier_setting=None, downscale=32, volume_type='score', **kwargs):
        return DataManager.get_warped_volume_basename(stack_m=stack, classifier_setting_m=classifier_setting,
        downscale=downscale, type_m=volume_type)

    @staticmethod
    def get_warped_volume_basename(stack_m, stack_f=None,
                                classifier_setting_m=None,
                                classifier_setting_f=None,
                                warp_setting=None,
                                downscale=32, type_f='score', type_m='score',
                                trial_idx=None, **kwargs):

        if classifier_setting_m is None:
            basename1 = '%(s1)s_down%(d1)d_%(t1)s' % \
            {'s1': stack_m,
            'd1': downscale,
            't1': volume_type_to_str(type_m)}
        else:
            basename1 = '%(s1)s_down%(d1)d_%(t1)s_clf_%(c1)d' % \
            {'s1': stack_m,
            'd1': downscale,
            'c1': classifier_setting_m,
            't1': volume_type_to_str(type_m)}

        if stack_f is not None:
            if classifier_setting_f is None:
                basename2 = \
                '%(s2)s_down%(d2)d_%(t2)s' % \
                {'s2': stack_f,
                'd2': downscale,
                't2': volume_type_to_str(type_f)}
            else:
                basename2 = \
                '%(s2)s_down%(d2)d_%(t2)s_clf_%(c2)d' % \
                {'s2': stack_f,
                'd2': downscale,
                'c2': classifier_setting_f,
                't2': volume_type_to_str(type_f)}

            basename = basename1 + '_warp_%(w)d_' % {'w': warp_setting} + basename2
        else:
            basename = basename1

            # basename = '%(s1)s_down%(d1)d_%(t1)s_clf_%(c1)d_warp_%(w)d_%(s2)s_down%(d2)d_%(t2)s_clf_%(c2)d' % \
            #   {'s1': stack_m, 's2': stack_f,
            #   'd1': downscale, 'd2': downscale,
            #   'c1': classifier_setting_m, 'c2': classifier_setting_f,
            #   'w': warp_setting,
            #   't1': volume_type_to_str(type_m), 't2': volume_type_to_str(type_f)}

        if trial_idx is not None:
            basename += '_trial_%d' % trial_idx

        return basename

    @staticmethod
    def get_alignment_parameters_filepath(stack_f, stack_m, warp_setting,
    classifier_setting_m=None, classifier_setting_f=None,
    type_f='score', type_m='score', param_suffix=None,
    downscale=32, trial_idx=0):
        basename = DataManager.get_warped_volume_basename(**locals())

        if param_suffix is None:
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m,
                    basename, basename + '_parameters.txt' % \
                    {'param_suffix':param_suffix})
        else:
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m,
                                basename, basename + '_parameters_%(param_suffix)s.txt' % \
                                {'param_suffix':param_suffix})

    @staticmethod
    def load_alignment_parameters(stack_f, stack_m, warp_setting,
    classifier_setting_m=None, classifier_setting_f=None,
    type_f='score', type_m='score', param_suffix=None,
    downscale=32, trial_idx=None):
        params_fp = DataManager.get_alignment_parameters_filepath(**locals())
        download_from_s3_to_ec2(params_fp)
        return DataManager.load_data(params_fp, 'transform_params')

    # @save_to_s3(fpkw='fp', fppos=0)
    @staticmethod
    def save_alignment_parameters(fp, params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f):

        create_if_not_exists(os.path.dirname(fp))
        with open(fp, 'w') as f:
            f.write(array_to_one_liner(params))
            f.write(array_to_one_liner(centroid_m))
            f.write(array_to_one_liner([xdim_m, ydim_m, zdim_m]))
            f.write(array_to_one_liner(centroid_f))
            f.write(array_to_one_liner([xdim_f, ydim_f, zdim_f]))

    @staticmethod
    def get_alignment_score_plot_filepath(stack_f, stack_m, warp_setting,
    classifier_setting_m=None, classifier_setting_f=None,
    type_f='score', type_m='score', param_suffix=None,
    downscale=32, trial_idx=None):
        basename = DataManager.get_warped_volume_basename(**locals())

        if param_suffix is None:
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m,
                            basename, basename + '_scoreEvolution.png')
        else:
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m,
                            basename, basename + '_scoreEvolution_%(param_suffix)s.png' % \
                            {'param_suffix': param_suffix})


    @staticmethod
    def get_score_history_filepath(stack_f, stack_m, warp_setting,
    classifier_setting_m=None, classifier_setting_f=None,
    type_f='score', type_m='score', param_suffix=None,
    downscale=32, trial_idx=0):
        basename = DataManager.get_warped_volume_basename(**locals())

        if param_suffix is None:
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m,
                    basename, basename + '_scoreHistory.bp' % \
                    {'param_suffix':param_suffix})
        else:
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m,
                                basename, basename + '_scoreHistory_%(param_suffix)s.bp' % \
                                {'param_suffix':param_suffix})

    # @staticmethod
    # def get_alignment_viz_dir(stack_m, stack_f,
    #                             classifier_setting_m,
    #                             classifier_setting_f,
    #                             warp_setting,
    #                             type_m='score', type_f='score',
    #                             downscale=32,
    #                             trial_idx=0):
    #
    #     basename = DataManager.get_warped_volume_basename(**locals())
    #     return os.path.join(REGISTRATION_VIZ_ROOTDIR, stack_m, basename)

    @staticmethod
    def get_alignment_viz_filepath(stack_m, stack_f,
                                classifier_setting_m,
                                classifier_setting_f,
                                warp_setting,
                                section,
                                type_m='score', type_f='score',
                                downscale=32,
                                trial_idx=0):

        basename = DataManager.get_warped_volume_basename(**locals())
        return os.path.join(REGISTRATION_VIZ_ROOTDIR, stack_m, basename, basename + '_%04d.jpg' % section)

    @staticmethod
    def load_confidence(stack_m, stack_f,
                            classifier_setting_m, classifier_setting_f, warp_setting, what,
                            type_m='score', type_f='score', param_suffix=None,
                            trial_idx=0):
        return load_pickle(DataManager.get_confidence_filepath(**locals()))

    @staticmethod
    def get_confidence_filepath(stack_m, stack_f,
                            classifier_setting_m, classifier_setting_f, warp_setting, what,
                            type_m='score', type_f='score', param_suffix=None,
                            trial_idx=0):
        basename = DataManager.get_warped_volume_basename(**locals())

        if param_suffix is None:
            fn = basename + '_parameters' % {'param_suffix':param_suffix}
        else:
            fn = basename + '_parameters_%(param_suffix)s' % {'param_suffix':param_suffix}

        if what == 'hessians':
            return os.path.join(REGISTRTION_PARAMETERS_ROOTDIR, stack_m, basename, fn + '_hessians.pkl')
        elif what == 'zscores':
            return os.path.join(REGISTRTION_PARAMETERS_ROOTDIR, stack_m, basename, fn + '_zscores.pkl')
        raise

    @staticmethod
    def get_classifier_filepath(structure, setting):
        clf_fp = os.path.join(CLF_ROOTDIR, 'setting_%(setting)s', 'classifiers', '%(structure)s_clf_setting_%(setting)d.dump') % {'structure': structure, 'setting':setting}
        return clf_fp

    @staticmethod
    def load_classifiers(setting, structures=all_known_structures):

        from sklearn.externals import joblib

        clf_allClasses = {}
        for structure in structures:
            clf_fp = DataManager.get_classifier_filepath(structure=structure, setting=setting)
            download_from_s3_to_ec2(clf_fp)
            if os.path.exists(clf_fp):
                clf_allClasses[structure] = joblib.load(clf_fp)
            else:
                sys.stderr.write('Setting %d: No classifier found for %s.\n' % (setting, structure))

        return clf_allClasses

    @staticmethod
    def load_sparse_scores(stack, structure, setting, sec=None, fn=None, anchor_fn=None):

        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        sparse_scores_fn = DataManager.get_sparse_scores_filepath(stack=stack, structure=structure,
                                            setting=setting, fn=fn, anchor_fn=anchor_fn)
        download_from_s3_to_ec2(sparse_scores_fn)
        return DataManager.load_data(sparse_scores_fn, filetype='bp')

    @staticmethod
    def get_sparse_scores_filepath(stack, structure, setting, sec=None, fn=None, anchor_fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        return os.path.join(SPARSE_SCORES_ROOTDIR, stack, '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped', \
                '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_sparseScores_setting_%(setting)s.hdf') % \
                {'fn': fn, 'anchor_fn': anchor_fn, 'structure':structure, 'setting': setting}

    @staticmethod
    def load_intensity_volume(stack, downscale=32):
        fn = DataManager.get_intensity_volume_filepath(stack=stack, downscale=downscale)
        return DataManager.load_data(fn, filetype='bp')

    @staticmethod
    def get_intensity_volume_filepath(stack, downscale=32):
        basename = DataManager.get_original_volume_basename(stack=stack, volume_type='intensity', downscale=downscale)
        vol_fn = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '.bp')
        return vol_fn

    @staticmethod
    def get_intensity_volume_bbox_filepath(stack, downscale=32):
        basename = DataManager.get_original_volume_basename(volume_type='intensity', **locals())
        return os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_bbox.txt')

    @staticmethod
    def load_annotation_as_score_volume(stack, downscale, structure):
        fn = DataManager.get_annotation_as_score_volume_filepath(**locals())
        return DataManager.load_data(fn, filetype='bp')

    @staticmethod
    def get_annotation_as_score_volume_filepath(stack, downscale, structure):
        basename = DataManager.get_original_volume_basename(volume_type='annotation_as_score', **locals())
        vol_fn = os.path.join(VOLUME_ROOTDIR, stack, basename, 'score_volumes', basename + '.bp')
        return vol_fn

    @staticmethod
    def load_annotation_volume(stack, downscale):
        fn = DataManager.get_annotation_volume_filepath(**locals())
        return DataManager.load_data(fn, filetype='bp')

    @staticmethod
    def get_annotation_volume_filepath(stack, downscale):
        basename = DataManager.get_original_volume_basename(volume_type='annotation', **locals())
        vol_fn = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '.bp')
        return vol_fn

    @staticmethod
    def get_annotation_volume_bbox_filepath(stack, downscale=32):
        basename = DataManager.get_original_volume_basename(volume_type='annotation', **locals())
        return os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_bbox.txt')

    @staticmethod
    def get_volume_label_to_name_filepath(stack):
        basename = DataManager.get_original_volume_basename(volume_type='annotation', **locals())
        fn = os.path.join(VOLUME_ROOTDIR, stack, basename, basename + '_nameToLabel.txt')
        # fn = os.path.join(volume_dir, stack, stack+'_down32_annotationVolume_nameToLabel.txt')
        return fn

    @staticmethod
    def load_volume_label_to_name(stack):
        fn = DataManager.get_volume_label_to_name_filepath(stack)
        label_to_name, name_to_label = DataManager.load_data(fn, filetype='label_name_map')
        return label_to_name, name_to_label

    ###################################
    # Mesh related
    ###################################

    @staticmethod
    def load_shell_mesh(stack, downscale, return_polydata_only=True):
        shell_mesh_fn = DataManager.get_shell_mesh_filepath(stack, downscale)
        return load_mesh_stl(shell_mesh_fn, return_polydata_only=return_polydata_only)

    @staticmethod
    def get_shell_mesh_filepath(stack, downscale):
        basename = DataManager.get_original_volume_basename(stack=stack, downscale=downscale, volume_type='outer_contour')
        shell_mesh_fn = os.path.join(MESH_ROOTDIR, stack, basename, basename + "_smoothed.stl")
        return shell_mesh_fn

    # @staticmethod
    # def load_meshes(stack, classifier_setting=None, structures=None, sided=False, return_polydata_only=True):
    #
    #     kwargs = locals()
    #
    #     if structures is None:
    #         if sided:
    #             structures = all_known_structures_sided
    #         else:
    #             structures = all_known_structures
    #
    #     meshes = {}
    #     for structure in structures:
    #         try:
    #             meshes[structure] = DataManager.load_mesh(structure=structure, **kwargs)
    #         except Exception as e:
    #             sys.stderr.write('%s\n' % e)
    #             sys.stderr.write('Error loading mesh for %s.\n' % structure)
    #
    #     return meshes



    # @staticmethod
    # def get_mesh_filepath(stack, structure, classifier_setting, downscale=32):
    #     basename = DataManager.get_original_volume_basename(stack=stack, downscale=downscale, classifier_setting=classifier_setting)
    #     fn = basename + '_%s' % structure
    #     mesh_fn = os.path.join(MESH_ROOTDIR, stack, basename, 'structure_mesh', fn + '.stl')
    #     print mesh_fn
    #     return mesh_fn

    # @staticmethod
    # def get_annotation_volume_mesh_filepath(stack, downscale, label):
    #     fn = os.path.join(MESH_ROOTDIR, stack, "%(stack)s_down%(ds)s_annotationVolume_%(name)s_smoothed.stl" % {'stack': stack, 'name': label, 'ds':downscale})
    #     return fn
    #
    # @staticmethod
    # def load_annotation_volume_mesh(stack, downscale, label, return_polydata_only=True):
    #     fn = DataManager.get_annotation_volume_mesh_filepath(stack, downscale, label)
    #     return load_mesh_stl(fn, return_polydata_only=return_polydata_only)


    # @staticmethod
    # def load_mesh(stack, structure, classifier_setting, return_polydata_only=True, **kwargs):
    #     mesh_fn = DataManager.get_mesh_filepath(stack=stack, structure=structure, classifier_setting=classifier_setting)
    #     mesh = load_mesh_stl(mesh_fn, return_polydata_only=return_polydata_only)
    #     if mesh is None:
    #         raise Exception('Mesh is empty.')
    #     return mesh

    @staticmethod
    def load_mesh(stack_m,
                                    structure,
                                    classifier_setting_m=None,
                                    stack_f=None,
                                    classifier_setting_f=None,
                                    warp_setting=None,
                                    downscale=32,
                                    type_m='score', type_f='score',
                                    trial_idx=0,
                                    return_polydata_only=True,
                                    **kwargs):
        mesh_fp = DataManager.get_mesh_filepath(**locals())
        mesh = load_mesh_stl(mesh_fp, return_polydata_only=return_polydata_only)
        if mesh is None:
            raise Exception('Mesh is empty: %s.' % structure)
        return mesh

    @staticmethod
    def load_mesh_atlasV2(stack_m,
                                    structure,
                                    classifier_setting_m=None,
                                    stack_f=None,
                                    classifier_setting_f=None,
                                    warp_setting=None,
                                    downscale=32,
                                    type_m='score', type_f='score',
                                    trial_idx=0,
                                    return_polydata_only=True,
                                    **kwargs):
        """
        For backward compatibility.
        """

        mesh_fp = DataManager.get_mesh_filepath_atlasV2(**locals())
        mesh = load_mesh_stl(mesh_fp, return_polydata_only=return_polydata_only)
        if mesh is None:
            raise Exception('Mesh is empty: %s.' % structure)
        return mesh

    @staticmethod
    def load_meshes(stack_m,
                                    stack_f=None,
                                    classifier_setting_m=None,
                                    classifier_setting_f=None,
                                    warp_setting=None,
                                    downscale=32,
                                    type_m='score', type_f='score',
                                    trial_idx=0,
                                    structures=None,
                                    sided=True,
                                    return_polydata_only=True):

        kwargs = locals()

        if structures is None:
            if sided:
                structures = all_known_structures_sided
            else:
                structures = all_known_structures

        meshes = {}
        for structure in structures:
            try:
                meshes[structure] = DataManager.load_mesh(structure=structure, **kwargs)
            except Exception as e:
                sys.stderr.write('%s\n' % e)
                sys.stderr.write('Error loading mesh for %s.\n' % structure)

        return meshes

    @staticmethod
    def load_meshes_atlasV2(stack_m,
                                    stack_f=None,
                                    classifier_setting_m=None,
                                    classifier_setting_f=None,
                                    warp_setting=None,
                                    downscale=32,
                                    type_m='score', type_f='score',
                                    trial_idx=0,
                                    structures=None,
                                    sided=True,
                                    return_polydata_only=True):
        """
        For backward compatibility.
        """

        kwargs = locals()

        if structures is None:
            if sided:
                structures = all_known_structures_sided
            else:
                structures = all_known_structures

        meshes = {}
        for structure in structures:
            try:
                meshes[structure] = DataManager.load_mesh_atlasV2(structure=structure, **kwargs)
            except Exception as e:
                sys.stderr.write('%s\n' % e)
                sys.stderr.write('Error loading mesh for %s.\n' % structure)

        return meshes

    # @staticmethod
    # def load_transformed_volume_meshes(stack_m,
    #                                 stack_f=None,
    #                                 classifier_setting_m=None,
    #                                 classifier_setting_f=None,
    #                                 warp_setting=None,
    #                                 downscale=32,
    #                                 type_m='score', type_f='score',
    #                                 trial_idx=0,
    #                                 structures=None,
    #                                 sided=False,
    #                                 return_polydata_only=True):
    #
    #     kwargs = locals()
    #
    #     if structures is None:
    #         if sided:
    #             structures = all_known_structures_sided
    #         else:
    #             structures = all_known_structures
    #
    #     meshes = {}
    #     for structure in structures:
    #         try:
    #             meshes[structure] = DataManager.load_transformed_volume_mesh(structure=structure, **kwargs)
    #         except Exception as e:
    #             sys.stderr.write('%s\n' % e)
    #             sys.stderr.write('Error loading mesh for %s.\n' % structure)
    #
    #     return meshes
    #
    # @staticmethod
    # def load_transformed_volume_mesh(stack_m,
    #                                 structure,
    #                                 classifier_setting_m=None,
    #                                 stack_f=None,
    #                                 classifier_setting_f=None,
    #                                 warp_setting=None,
    #                                 downscale=32,
    #                                 type_m='score', type_f='score',
    #                                 trial_idx=0,
    #                                 return_polydata_only=True,
    #                                 **kwargs):
    #     mesh_fp = DataManager.get_transformed_volume_mesh_filepath(**locals())
    #     mesh = load_mesh_stl(mesh_fp, return_polydata_only=return_polydata_only)
    #     if mesh is None:
    #         raise Exception('Mesh is empty: %s.' % structure)
    #     return mesh
    #
    #
    # @staticmethod
    # def get_original_volume_mesh_filepath(stack_m,
    #                                         structure,
    #                                         classifier_setting_m=None,
    #                                         classifier_setting_f=None,
    #                                         warp_setting=None,
    #                                         stack_f=None,
    #                                         downscale=32,
    #                                         type_m='score', type_f='score',
    #                                         trial_idx=0, **kwargs):
    #     basename = DataManager.get_warped_volume_basename(**locals())
    #     fn = basename + '_%s' % structure
    #     return os.path.join(MESH_ROOTDIR, stack_m, basename, fn + '.stl')

    # @staticmethod
    # def get_mesh_filepath(stack_m,
    #                                         structure,
    #                                         classifier_setting_m=None,
    #                                         classifier_setting_f=None,
    #                                         warp_setting=None,
    #                                         stack_f=None,
    #                                         downscale=32,
    #                                         type_m='score', type_f='score',
    #                                         trial_idx=0, **kwargs):
    #     basename = DataManager.get_warped_volume_basename(**locals())
    #     fn = basename + '_%s' % structure
    #     return os.path.join(MESH_ROOTDIR, stack_m, basename, fn + '.stl')

    @staticmethod
    def get_mesh_filepath(stack_m,
                                            structure,
                                            classifier_setting_m=None,
                                            classifier_setting_f=None,
                                            warp_setting=None,
                                            stack_f=None,
                                            downscale=32,
                                            type_m='score', type_f='score',
                                            trial_idx=0, **kwargs):
        basename = DataManager.get_warped_volume_basename(**locals())
        fn = basename + '_%s' % structure
        return os.path.join(MESH_ROOTDIR, stack_m, basename, fn + '.stl')

    @staticmethod
    def get_mesh_filepath_atlasV2(stack_m,
                                            structure,
                                            classifier_setting_m=None,
                                            classifier_setting_f=None,
                                            warp_setting=None,
                                            stack_f=None,
                                            downscale=32,
                                            type_m='score', type_f='score',
                                            trial_idx=0, **kwargs):
        """
        For backward compatibility.
        """
        basename = DataManager.get_warped_volume_basename(**locals())
        fn = basename + '_%s' % structure
        return os.path.join(MESH_ROOTDIR, stack_m, basename, 'structure_mesh', fn + '.stl')

    # @staticmethod
    # def get_transformed_volume_mesh_filepath(stack_m,
    #                                         structure,
    #                                         classifier_setting_m=None,
    #                                         classifier_setting_f=None,
    #                                         warp_setting=None,
    #                                         stack_f=None,
    #                                         downscale=32,
    #                                         type_m='score', type_f='score',
    #                                         trial_idx=0, **kwargs):
    #     basename = DataManager.get_warped_volume_basename(**locals())
    #     fn = basename + '_%s' % structure
    #     return os.path.join(MESH_ROOTDIR, stack_m, basename, fn + '.stl')

    @staticmethod
    def load_volume_all_known_structures(stack_m, stack_f,
                                        warp_setting,
                                        classifier_setting_m=None,
                                        classifier_setting_f=None,
                                        type_m='score',
                                        type_f='score',
                                        downscale=32,
                                        structures=None,
                                        trial_idx=0,
                                        sided=True,
                                        include_surround=False):
        if stack_f is not None:
            return DataManager.load_transformed_volume_all_known_structures(**locals())
        else:
            raise Exception('Not implemented.')

    # @staticmethod
    # def load_volume(stack_m, stack_f=None,
    #                 warp_setting=None,
    #                 classifier_setting_m=None,
    #                 classifier_setting_f=None,
    #                 type_m='score',
    #                  type_f='score',
    #                 structure=None,
    #                 downscale=32,
    #                 trial_idx=0):
    #     if stack_f is not None:
    #         return DataManager.load_transformed_volume(**locals())
    #     elif type_m == 'score':
    #         DataManager.get_original_volume_filepath
    #     else:
    #         raise Exception('Not implemented.')

    @staticmethod
    def load_transformed_volume(stack_m, stack_f,
                                        warp_setting,
                                        classifier_setting_m=None,
                                        classifier_setting_f=None,
                                        type_m='score',
                                         type_f='score',
                                        structure=None,
                                        downscale=32,
                                        trial_idx=0):
        fp = DataManager.get_transformed_volume_filepath(**locals())
        download_from_s3_to_ec2(fp)
        return DataManager.load_data(fp, filetype='bp')

    @staticmethod
    def load_transformed_volume_all_known_structures(stack_m, stack_f,
                                        warp_setting,
                                        classifier_setting_m=None,
                                        classifier_setting_f=None,
                                        type_m='score',
                                        type_f='score',
                                        downscale=32,
                                        structures=None,
                                        trial_idx=0,
                                        sided=True,
                                        include_surround=False):
        if structures is None:
            if sided:
                if include_surround:
                    structures = all_known_structures_sided_with_surround
                else:
                    structures = all_known_structures_sided
            else:
                structures = all_known_structures

        volumes = {}
        for structure in structures:
            try:
                volumes[structure] = DataManager.load_transformed_volume(stack_m=stack_m, type_m=type_m,
                                                    stack_f=stack_f, type_f=type_f, downscale=downscale,
                                                    classifier_setting_m=classifier_setting_m,
                                                    classifier_setting_f=classifier_setting_f,
                                                    warp_setting=warp_setting,
                                                    structure=structure,
                                                    trial_idx=trial_idx)
            except Exception as e:
                sys.stderr.write('%s\n' % e)
                sys.stderr.write('Score volume for %s does not exist.\n' % structure)
        return volumes

    @staticmethod
    def get_transformed_volume_filepath(stack_m, stack_f,
                                        warp_setting,
                                        classifier_setting_m=None,
                                        classifier_setting_f=None,
                                        downscale=32,
                                         type_m='score',
                                          type_f='score',
                                        structure=None,
                                        trial_idx=0):

        basename = DataManager.get_warped_volume_basename(**locals())

        if structure is not None:
            fn = basename + '_%s' % structure

        return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'score_volumes', fn + '.bp')


    @staticmethod
    def load_prob_shapes(stack_m, stack_f=None,
            classifier_setting_m=None,
            classifier_setting_f=None,
            warp_setting=None,
            downscale=32,
            type_m='score', type_f='score',
            trial_idx=0,
            structures=None,
            sided=True,
            return_polydata_only=True):

        kwargs = locals()

        if structures is None:
            if sided:
                structures = all_known_structures_sided
            else:
                structures = all_known_structures

        prob_shapes = {}
        for structure in structures:
            try:
                vol = bp.unpack_ndarray_file(DataManager.get_prob_shape_volume_filepath(structure=structure, **kwargs))
                origin = np.loadtxt(DataManager.get_prob_shape_origin_filepath(structure=structure, **kwargs))
                prob_shapes[structure] = (vol, origin)
            except Exception as e:
                sys.stderr.write('%s\n' % e)
                sys.stderr.write('Error loading probablistic shape for %s.\n' % structure)

        return prob_shapes

    @staticmethod
    def get_prob_shape_viz_filepath(stack_m, stack_f=None,
                                        warp_setting=None,
                                        classifier_setting_m=None,
                                        classifier_setting_f=None,
                                        downscale=32,
                                         type_m='score',
                                         type_f='score',
                                        structure=None,
                                        trial_idx=0,
                                        suffix=None,
                                        **kwargs):
        """
        Return prob. shape volume filepath.
        """

        basename = DataManager.get_warped_volume_basename(**locals())

        assert structure is not None
        fn = basename + '_' + structure + '_' + suffix

        return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'probabilistic_shape_viz', structure, fn + '.png')

    @staticmethod
    def get_prob_shape_volume_filepath(stack_m, stack_f=None,
                                        warp_setting=None,
                                        classifier_setting_m=None,
                                        classifier_setting_f=None,
                                        downscale=32,
                                         type_m='score',
                                         type_f='score',
                                        structure=None,
                                        trial_idx=0, **kwargs):
        """
        Return prob. shape volume filepath.
        """

        basename = DataManager.get_warped_volume_basename(**locals())
        if structure is not None:
            fn = basename + '_' + structure

        return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'probabilistic_shapes', fn + '.bp')

    @staticmethod
    def get_prob_shape_origin_filepath(stack_m, stack_f=None,
                                        warp_setting=None,
                                        classifier_setting_m=None,
                                        classifier_setting_f=None,
                                        downscale=32,
                                         type_m='score',
                                         type_f='score',
                                        structure=None,
                                        trial_idx=0, **kwargs):
        """
        Return prob. shape volume origin filepath.

        Note that these origins are with respect to

        """

        basename = DataManager.get_warped_volume_basename(**locals())
        if structure is not None:
            fn = basename + '_' + structure
        return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'probabilistic_shapes', fn + '_origin.txt')

    @staticmethod
    def get_volume_filepath(stack_m, stack_f=None,
                                        warp_setting=None,
                                        classifier_setting_m=None,
                                        classifier_setting_f=None,
                                        downscale=32,
                                         type_m='score',
                                          type_f='score',
                                        structure=None,
                                        trial_idx=0):

        basename = DataManager.get_warped_volume_basename(**locals())

        if structure is not None:
            fn = basename + '_' + structure

        if type_m == 'score':
            return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'score_volumes', fn + '.bp')
        else:
            raise


    @staticmethod
    def get_score_volume_filepath(stack, structure, volume_type='score', downscale=32, classifier_setting=None):

        basename = DataManager.get_original_volume_basename(**locals())
        vol_fn = os.path.join(VOLUME_ROOTDIR, stack, basename, 'score_volumes', basename + '_' + structure + '.bp')
        return vol_fn

    @staticmethod
    def get_volume_gradient_filepath_template_scratch(stack, structure, downscale=32, classifier_setting=None, volume_type='score', **kwargs):
        basename = DataManager.get_original_volume_basename(**locals())
        grad_fn = os.path.join('/scratch', 'CSHL_volumes', stack, basename, 'score_volume_gradients', basename + '_' + structure + '_%(suffix)s.bp')
        return grad_fn


    @staticmethod
    def get_volume_gradient_filepath_scratch(stack, structure, suffix, volume_type='score', classifier_setting=None, downscale=32):
        grad_fn = DataManager.get_volume_gradient_filepath_template_scratch(**locals())  % {'suffix': suffix}
        return grad_fn


    @staticmethod
    def get_volume_gradient_filepath_template(stack, structure, downscale=32, classifier_setting=None, volume_type='score', **kwargs):
        basename = DataManager.get_original_volume_basename(**locals())
        grad_fn = os.path.join(VOLUME_ROOTDIR, stack, basename, 'score_volume_gradients', basename + '_' + structure + '_%(suffix)s.bp')
        return grad_fn

    @staticmethod
    def get_volume_gradient_filepath(stack, structure, suffix, volume_type='score', classifier_setting=None, downscale=32):
        grad_fn = DataManager.get_volume_gradient_filepath_template(**locals())  % {'suffix': suffix}
        return grad_fn

    # @staticmethod
    # def get_score_volume_gradient_filepath_template(stack, structure, downscale=32, classifier_setting=None, volume_type='score'):
    #     basename = DataManager.get_original_volume_basename(**locals())
    #     grad_fn = os.path.join(VOLUME_ROOTDIR, stack, basename, 'score_volume_gradients', basename + '_' + structure + '_%(suffix)s.bp')
    #     return grad_fn

    # @staticmethod
    # def get_score_volume_gradient_filepath(stack, structure, suffix, volume_type='score', classifier_setting=None, downscale=32):
    #     grad_fn = DataManager.get_score_volume_gradient_filepath_template(stack=stack, structure=structure,
    #                                         classifier_setting=classifier_setting, downscale=downscale, volume_type=volume_type) % \
    #                                         {'suffix': suffix}
    #     return grad_fn

    @staticmethod
    def load_original_volume_all_known_structures(stack, downscale=32, classifier_setting=None, structures=None, sided=True, volume_type='score',
                                                return_structure_index_mapping=True):

        if structures is None:
            if sided:
                structures = all_known_structures_sided
            else:
                structures = all_known_structures

        if return_structure_index_mapping:

            try:
                label_to_structure, structure_to_label = DataManager.load_volume_label_to_name(stack=stack)
                loaded = True
                sys.stderr.write('Load structure/index map.\n')
            except:
                loaded = False
                sys.stderr.write('Prior structure/index map not found. Generating a new one.\n')

            volumes = {}
            if not loaded:
                structure_to_label = {}
                label_to_structure = {}
                index = 1
            for structure in sorted(structures):
                try:
                    if loaded:
                        index = structure_to_label[structure]

                    volumes[index] = DataManager.load_original_volume(stack=stack, structure=structure,
                                            downscale=downscale, classifier_setting=classifier_setting,
                                            volume_type=volume_type)
                    if not loaded:
                        structure_to_label[structure] = index
                        label_to_structure[index] = structure
                        index += 1
                except:
                    sys.stderr.write('Score volume for %s does not exist.\n' % structure)

            # One volume at down=32 takes about 1MB of memory.

            sys.stderr.write('Volume shape: (%d, %d, %d)\n' % volumes.values()[0].shape)
            return volumes, structure_to_label, label_to_structure

        else:
            volumes = {}
            for structure in structures:
                try:
                    volumes[structure] = DataManager.load_original_volume(stack=stack, structure=structure,
                                            downscale=downscale, classifier_setting=classifier_setting,
                                            volume_type=volume_type)

                except:
                    sys.stderr.write('Score volume for %s does not exist.\n' % structure)

            sys.stderr.write('Volume shape: (%d, %d, %d)\n' % volumes.values()[0].shape)
            return volumes


    @staticmethod
    def get_original_volume_filepath(stack, structure, volume_type='score', downscale=32, classifier_setting=None):
        if volume_type == 'score':
            fp = DataManager.get_score_volume_filepath(**locals())
        elif volume_type == 'annotation':
            fp = DataManager.get_annotation_volume_filepath(stack=stack, downscale=downscale)
        elif volume_type == 'intensity':
            fp = DataManager.get_intensity_volume_filepath(stack=stack, downscale=downscale)
        else:
            raise Exception("Volume type must be one of score, annotation or intensity.")
        return fp

    @staticmethod
    def load_original_volume(stack, structure, downscale, classifier_setting=None, volume_type='score'):
        vol_fp = DataManager.get_original_volume_filepath(**locals())
        download_from_s3_to_ec2(vol_fp, is_dir=False)
        volume = DataManager.load_data(vol_fp, filetype='bp')
        return volume

#     @staticmethod
#     def load_score_volume(stack, structure, downscale, classifier_setting=None, volume_type='score'):
#         # vol_fn = DataManager.get_score_volume_filepath(**locals())
#         vol_fn = DataManager.get_volume_filepath(stack_m=stack, structure=structure, downscale=downscale, classifier_setting_m=classifier_setting, type_m=volume_type, trial_idx=None)

#         download_from_s3_to_ec2(vol_fn, is_dir=False)
#         score_volume = DataManager.load_data(vol_fn, filetype='bp')
#         return score_volume

    @staticmethod
    def load_original_volume_bbox(stack, vol_type, classifier_setting=None, structure=None, downscale=32):
        """
        This returns the 3D bounding box.

        Args:
            type (str):
                annotation: with respect to aligned uncropped thumbnail
                score/thumbnail: with respect to aligned cropped thumbnail
                shell: with respect to aligned uncropped thumbnail
        """
        bbox_fp = DataManager.get_original_volume_bbox_filepath(**locals())
        download_from_s3_to_ec2(bbox_fp)
        volume_bbox = DataManager.load_data(bbox_fp, filetype='bbox')
        return volume_bbox


    @staticmethod
    def get_original_volume_bbox_filepath(stack,
                                classifier_setting,
                                downscale=32,
                                 vol_type='score',
                                structure=None):
        if vol_type == 'annotation':
            bbox_fn = DataManager.get_annotation_volume_bbox_filepath(stack=stack)
        elif vol_type == 'score':
            bbox_fn = DataManager.get_score_volume_bbox_filepath(stack=stack, structure=structure, downscale=downscale,
            classifier_setting=classifier_setting)
        elif vol_type == 'shell':
            bbox_fn = DataManager.get_shell_bbox_filepath(stack, structure, downscale)
        elif vol_type == 'thumbnail':
            bbox_fn = DataManager.get_score_volume_bbox_filepath(stack=stack, structure='7N', downscale=downscale,
            classifier_setting=classifier_setting)
        else:
            raise Exception('Type must be annotation, score, shell or thumbnail.')

        return bbox_fn

    @staticmethod
    def get_shell_bbox_filepath(stack, label, downscale):
        bbox_filepath = VOLUME_ROOTDIR + '/%(stack)s/%(stack)s_down%(ds)d_outerContourVolume_bbox.txt' % \
                        dict(stack=stack, ds=downscale)
        return bbox_filepath

    @staticmethod
    def get_score_volume_bbox_filepath(stack, structure, downscale, classifier_setting):
        # Volume bounding box is independent of classifier setting.
        basename = DataManager.get_original_volume_basename(stack=stack, downscale=downscale, classifier_setting=classifier_setting)
        score_volume_bbox_filepath = os.path.join(VOLUME_ROOTDIR,  stack, basename, 'score_volumes', \
                                    basename + '_%(structure)s_bbox.txt' % dict(structure=structure))
        return score_volume_bbox_filepath

    # @staticmethod
    # def get_scoremap_viz_filepath(stack, section=None, fn=None, anchor_fn=None, structure=None, setting=None):
    #
    #     if section is not None:
    #         fn = metadata_cache['sections_to_filenames'][stack][section]
    #         if is_invalid(fn): raise Exception('Section is invalid: %s.' % fn)
    #
    #     if anchor_fn is None:
    #         anchor_fn = metadata_cache['anchor_fn'][stack]
    #
    #     scoremap_viz_filepath = os.path.join(SCOREMAP_VIZ_ROOTDIR, '%(structure)s/%(stack)s/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_denseScoreMap_viz_setting_%(setting)d.jpg') \
    #         % {'stack': stack, 'fn': fn, 'structure': structure, 'anchor_fn': anchor_fn, 'setting': setting}
    #
    #     return scoremap_viz_filepath

    # @staticmethod
    # def get_downscaled_scoremap_viz_filepath(stack, section=None, fn=None, anchor_fn=None, structure=None, setting=None,
    # downscale=32):
    #
    #     if section is not None:
    #         fn = metadata_cache['sections_to_filenames'][stack][section]
    #         if is_invalid(fn): raise Exception('Section is invalid: %s.' % fn)
    #
    #     if anchor_fn is None:
    #         anchor_fn = metadata_cache['anchor_fn'][stack]
    #
    #     viz_dir = os.path.join(ROOT_DIR, 'CSHL_scoremaps_down%(down)d_viz' % dict(down=downscale))
    #     basename = '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(down)d' % dict(fn=fn, anchor_fn=anchor_fn, down=downscale)
    #     fn = basename + '_%(structure)s_denseScoreMap_setting_%(setting)d.jpg' % dict(structure=structure, setting=setting)
    #
    #     scoremap_viz_filepath = os.path.join(viz_dir, structure, stack, fn)
    #
    #     return scoremap_viz_filepath

    @staticmethod
    def get_scoremap_viz_filepath(stack, downscale, section=None, fn=None, anchor_fn=None, structure=None, setting=None):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn): raise Exception('Section is invalid: %s.' % fn)

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        viz_dir = os.path.join(ROOT_DIR, 'CSHL_scoremaps_down%(down)d_viz' % dict(down=downscale))
        basename = '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(down)d' % dict(fn=fn, anchor_fn=anchor_fn, down=downscale)
        fn = basename + '_%(structure)s_denseScoreMap_setting_%(setting)d.jpg' % dict(structure=structure, setting=setting)

        scoremap_viz_filepath = os.path.join(viz_dir, structure, stack, fn)

        return scoremap_viz_filepath

    @staticmethod
    def get_downscaled_scoremap_filepath(stack, structure, setting, downscale, section=None, fn=None, anchor_fn=None, return_bbox_fp=False):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn):
                raise Exception('Section is invalid: %s.' % fn)

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        basename = '%(fn)s_alignedTo_%(anchor_fn)s_cropped_down%(down)d' % dict(fn=fn, anchor_fn=anchor_fn, down=downscale)

        scoremap_bp_filepath = os.path.join(ROOT_DIR, 'CSHL_scoremaps_down%(down)d' % dict(down=downscale), stack, basename,
        basename + '_%(structure)s_denseScoreMap_setting_%(setting)d.bp' % dict(structure=structure, setting=setting))

        return scoremap_bp_filepath

    @staticmethod
    def load_downscaled_scoremap(stack, structure, setting, section=None, fn=None, anchor_fn=None, downscale=32):
        """
        Return scoremaps as bp files.
        """

        # Load scoremap
        scoremap_bp_filepath = DataManager.get_downscaled_scoremap_filepath(stack, section=section, \
                        fn=fn, anchor_fn=anchor_fn, structure=structure, setting=setting,
                        downscale=downscale)

        download_from_s3_to_ec2(scoremap_bp_filepath)

        if not os.path.exists(scoremap_bp_filepath):
            raise Exception('No scoremap for image %s (section %d) for label %s\n' % \
            (metadata_cache['sections_to_filenames'][stack][section], section, structure))

        scoremap_downscaled = DataManager.load_data(scoremap_bp_filepath, filetype='bp')
        return scoremap_downscaled

    @staticmethod
    def get_scoremap_filepath(stack, structure, setting, section=None, fn=None, anchor_fn=None, return_bbox_fp=False):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn):
                raise Exception('Section is invalid: %s.' % fn)

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        scoremap_bp_filepath = os.path.join(SCOREMAPS_ROOTDIR, stack, \
        '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_denseScoreMap_setting_%(setting)d.hdf') \
        % dict(stack=stack, fn=fn, structure=structure, anchor_fn=anchor_fn, setting=setting)

        scoremap_bbox_filepath = os.path.join(SCOREMAPS_ROOTDIR, stack, \
        '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_denseScoreMap_interpBox.txt') \
            % dict(stack=stack, fn=fn, structure=structure, anchor_fn=anchor_fn)

        if return_bbox_fp:
            return scoremap_bp_filepath, scoremap_bbox_filepath
        else:
            return scoremap_bp_filepath

    @staticmethod
    def load_scoremap(stack, structure, setting, section=None, fn=None, anchor_fn=None, downscale=1):
        """
        Return scoremaps.
        """

        # Load scoremap
        scoremap_bp_filepath, scoremap_bbox_filepath = DataManager.get_scoremap_filepath(stack, section=section, \
                                    fn=fn, anchor_fn=anchor_fn, structure=structure, return_bbox_fp=True, setting=setting)
        if not os.path.exists(scoremap_bp_filepath):
            raise Exception('No scoremap for image %s (section %d) for label %s\n' % \
            (metadata_cache['sections_to_filenames'][stack][section], section, structure))

        scoremap = DataManager.load_data(scoremap_bp_filepath, filetype='hdf')

        # Load interpolation box
        xmin, xmax, ymin, ymax = DataManager.load_data(scoremap_bbox_filepath, filetype='bbox')
        ymin_downscaled = ymin / downscale
        xmin_downscaled = xmin / downscale

        full_width, full_height = metadata_cache['image_shape'][stack]
        scoremap_downscaled = np.zeros((full_height/downscale, full_width/downscale), np.float32)

        # To conserve memory, it is important to make a copy of the sub-scoremap and delete the original scoremap
        scoremap_roi_downscaled = scoremap[::downscale, ::downscale].copy()
        del scoremap

        h_downscaled, w_downscaled = scoremap_roi_downscaled.shape

        scoremap_downscaled[ymin_downscaled : ymin_downscaled + h_downscaled,
                            xmin_downscaled : xmin_downscaled + w_downscaled] = scoremap_roi_downscaled

        return scoremap_downscaled


    # @staticmethod
    # def load_scoremap(stack, section=None, fn=None, anchor_fn=None, label=None, downscale_factor=32, train_sample_scheme=None):
    #     """
    #     Return scoremaps.
    #     """
    #
    #     # Load scoremap
    #
    #     scoremap_bp_filepath, scoremap_bbox_filepath = DataManager.get_scoremap_filepath(stack, section=section, \
    #                                                 fn=fn, anchor_fn=anchor_fn, label=label, return_bbox_fp=True, train_sample_scheme=train_sample_scheme)
    #     if not os.path.exists(scoremap_bp_filepath):
    #         raise Exception('No scoremap for section %d for label %s\n' % (section, label))
    #         # return None
    #     scoremap = DataManager.load_data(scoremap_bp_filepath, filetype='hdf')
    #
    #     # Load interpolation box
    #     xmin, xmax, ymin, ymax = DataManager.load_data(scoremap_bbox_filepath, filetype='bbox')
    #     ymin_downscaled = ymin / downscale_factor
    #     xmin_downscaled = xmin / downscale_factor
    #
    #     # full_width, full_height = DataManager.get_image_dimension(stack)
    #     # full_width, full_height = (16000, 13120)
    #     full_width, full_height = metadata_cache['image_shape'][stack]
    #     scoremap_downscaled = np.zeros((full_height/downscale_factor, full_width/downscale_factor), np.float32)
    #
    #     # To conserve memory, it is important to make a copy of the sub-scoremap and delete the original scoremap
    #     scoremap_roi_downscaled = scoremap[::downscale_factor, ::downscale_factor].copy()
    #     del scoremap
    #
    #     h_downscaled, w_downscaled = scoremap_roi_downscaled.shape
    #
    #     scoremap_downscaled[ymin_downscaled : ymin_downscaled + h_downscaled,
    #                         xmin_downscaled : xmin_downscaled + w_downscaled] = scoremap_roi_downscaled
    #
    #     return scoremap_downscaled

    @staticmethod
    def load_dnn_feature_locations(stack, model_name, section=None, fn=None, anchor_fn=None):
        fp = DataManager.get_dnn_feature_locations_filepath(stack=stack, model_name=model_name, section=section, fn=fn, anchor_fn=anchor_fn)
        download_from_s3_to_ec2(fp)
        locs = np.loadtxt(fp).astype(np.int)
        indices = locs[:, 0]
        locations = locs[:, 1:]
        return indices, locations

    @staticmethod
    def get_dnn_feature_locations_filepath(stack, model_name, section=None, fn=None, anchor_fn=None):

        if section is not None:
            section_to_filename = metadata_cache['sections_to_filenames'][stack]
            fn = section_to_filename[section]

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        feature_locs_fn = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack, \
        '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_patch_locations.txt' % \
        dict(fn=fn, anchor_fn=anchor_fn))

        # output_dir = create_if_not_exists(os.path.join(PATCH_FEATURES_ROOTDIR, stack,
        #                                '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped' % dict(fn=fn, anchor_fn=anchor_fn)))
        # output_indices_fn = os.path.join(output_dir,
        #                                  '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_patch_locations.txt' % \
        #                                  dict(fn=fn, anchor_fn=anchor_fn))
        return feature_locs_fn

    @staticmethod
    def get_dnn_features_filepath(stack, model_name, section=None, fn=None, anchor_fn=None):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        feature_fn = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack, \
        '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_features.hdf' % \
        dict(fn=fn, anchor_fn=anchor_fn))
        # feature_fn = PATCH_FEATURES_ROOTDIR + '/%(stack)s/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_features.hdf' % dict(stack=stack, fn=fn, anchor_fn=anchor_fn)
        return feature_fn

    @staticmethod
    def load_dnn_features(stack, model_name, section=None, fn=None, anchor_fn=None):
        features_fn = DataManager.get_dnn_features_filepath(stack=stack, model_name=model_name, section=section, fn=fn, anchor_fn=anchor_fn)
        download_from_s3_to_ec2(features_fn)
        return load_hdf(features_fn)

    @staticmethod
    def get_image_dir(stack, version='compressed', resol='lossless', anchor_fn=None, modality='nissl'):
        """
        resol: can be either lossless or thumbnail.
        version:
        - compressed: for regular nissl, RGB JPEG; for neurotrace, blue channel as grey JPEG
        - saturation: for regular nissl, saturation as gray, tif; for NT, blue channel as grey, tif
        - cropped: for regular nissl, lossless RGB tif; for NT, 16 bit, all channels (?) tif.
        """

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack)

        if resol == 'lossless' and version == 'compressed':
            image_dir = os.path.join(DATA_DIR, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn})
        # elif resol == 'lossless' and version == 'saturation':
        #     image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_saturation' % {'anchor_fn':anchor_fn})
        elif resol == 'lossless' and (version == 'cropped' or version == 'cropped_8bit'):
            if modality == 'fluorescent':
                image_dir = os.path.join(DATA_DIR, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_contrast_stretched' % {'anchor_fn':anchor_fn})
            else:
                image_dir = os.path.join(DATA_DIR, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
        elif resol == 'lossless' and version == 'cropped_16bit':
            image_dir = os.path.join(DATA_DIR, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
        elif resol == 'lossless' and version == 'cropped_8bit_blueasgray':
            image_dir = os.path.join(DATA_DIR, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_contrast_stretched_blueasgray' % {'anchor_fn':anchor_fn})
        elif resol == 'thumbnail' and version == 'cropped_tif':
            image_dir = os.path.join(THUMBNAIL_DATA_DIR, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
        elif (resol == 'thumbnail' and version == 'aligned') or (resol == 'thumbnail' and version == 'aligned_tif'):
            image_dir = os.path.join(THUMBNAIL_DATA_DIR, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s' % {'anchor_fn':anchor_fn})
        else:
            sys.stderr.write('Version %s and resolution %s not recognized.\n' % (version, resol))

        return image_dir


    @staticmethod
    def get_image_filepath(stack, section=None, version='compressed', resol='lossless', data_dir=DATA_DIR, fn=None, anchor_fn=None):
        """
        resol: can be either lossless or thumbnail.
        version:
        - compressed: for regular nissl, RGB JPEG; for neurotrace, blue channel as grey JPEG
        - saturation: for regular nissl, saturation as gray, tif; for NT, blue channel as grey, tif
        - cropped: for regular nissl, lossless RGB tif; for NT, 16 bit, all channels (?) tif.
        """

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn):
                raise Exception('Section is invalid: %s.' % fn)
        else:
            assert fn is not None

        is_fluorescent = (stack in all_ntb_stacks or stack in all_alt_nissl_ntb_stacks or stack in all_alt_nissl_tracing_stacks) and fn.split('-')[1][0] == 'F'

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack)

        if resol == 'thumbnail' and version == 'original_png':
            image_path = os.path.join(RAW_DATA_DIR, stack, fn + '.png')
        elif resol == 'lossless' and (version == 'compressed' or version == 'contrast_stretched_compressed' or version == 'cropped_compressed'):
            if is_fluorescent:
                image_dir = DataManager.get_image_dir(stack=stack, version=version, resol=resol, modality='fluorescent')
                # image_dir = os.path.join(DATA_DIR, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn})
                image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_contrast_stretched_compressed' % {'anchor_fn':anchor_fn}])
                image_path = os.path.join(image_dir, image_name + '.jpg')
            else:
                image_dir = DataManager.get_image_dir(stack=stack, version=version, resol=resol, modality='nissl')
                image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn}])
                image_path = os.path.join(image_dir, image_name + '.jpg')
        # elif resol == 'lossless' and version == 'saturation':
        #     if is_fluorescent:
        #         image_dir = os.path.join(DATA_DIR, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_blueAsGrayscale' % {'anchor_fn':anchor_fn})
        #         image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_blueAsGrayscale' % {'anchor_fn':anchor_fn}])
        #         image_path = os.path.join(image_dir, image_name + '.tif')
        #     else: # Nissl
        #         image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_saturation' % {'anchor_fn':anchor_fn})
        #         image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_saturation' % {'anchor_fn':anchor_fn}])
        #         image_path = os.path.join(image_dir, image_name + '.tif')
        elif resol == 'lossless' and (version == 'cropped' or version == 'cropped_8bit'):
            # image_dir = os.path.join(DATA_DIR, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
            if is_fluorescent:
                image_dir = DataManager.get_image_dir(stack=stack, version=version, resol=resol, modality='fluorescent')
                image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_contrast_stretched' % {'anchor_fn':anchor_fn}])
                image_path = os.path.join(image_dir, image_name + '.tif')
            else:
                image_dir = DataManager.get_image_dir(stack=stack, version=version, resol=resol, modality='nissl')
                image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn}])
                image_path = os.path.join(image_dir, image_name + '.tif')
        elif resol == 'lossless' and version == 'cropped_16bit':
            assert is_fluorescent
            image_dir = DataManager.get_image_dir(stack=stack, version=version, resol=resol, modality='fluorescent')
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn}])
            image_path = os.path.join(image_dir, image_name + '.tif')
        elif resol == 'lossless' and version == 'cropped_8bit_blueasgray':
            assert is_fluorescent
            image_dir = DataManager.get_image_dir(stack=stack, version=version, resol=resol, modality='fluorescent')
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_contrast_stretched_blueasgray' % {'anchor_fn':anchor_fn}])
            image_path = os.path.join(image_dir, image_name + '.tif')
        elif resol == 'thumbnail' and version == 'cropped':
            image_dir = os.path.join(THUMBNAIL_DATA_DIR, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn}])
            image_path = os.path.join(image_dir, image_name + '.tif')
        elif resol == 'thumbnail' and version == 'cropped_tif':
            image_dir = os.path.join(THUMBNAIL_DATA_DIR, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn}])
            image_path = os.path.join(image_dir, image_name + '.tif')
        elif (resol == 'thumbnail' and version == 'aligned') or (resol == 'thumbnail' and version == 'aligned_tif'):
            image_dir = os.path.join(THUMBNAIL_DATA_DIR, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s' % {'anchor_fn':anchor_fn})
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s' % {'anchor_fn':anchor_fn}])
            image_path = os.path.join(image_dir, image_name + '.tif')
        else:
            sys.stderr.write('Version %s and resolution %s not recognized.\n' % (version, resol))

        return image_path

    @staticmethod
    def get_annotated_structures(stack):
        """
        Return existing structures on every section in annotation.
        """
        contours, _ = load_annotation_v3(stack, annotation_rootdir=ANNOTATION_ROOTDIR)
        annotated_structures = {sec: list(set(contours[contours['section']==sec]['name']))
                                for sec in range(first_sec, last_sec+1)}
        return annotated_structures

    @staticmethod
    def load_annotation_v3(stack=None, annotation_rootdir=ANNOTATION_ROOTDIR):
        fn = os.path.join(annotation_rootdir, stack, '%(stack)s_annotation_v3.h5' % {'stack':stack})
        contour_df = DataManager.load_data(fn, filetype='annotation_hdf')

        try:
            structure_df = read_hdf(fn, 'structures')
        except Exception as e:
            print e
            sys.stderr.write('Annotation has no structures.\n')
            return contour_df, None

        sys.stderr.write('Loaded annotation %s.\n' % fp)
        return contour_df, structure_df

    @staticmethod
    def get_image_dimension(stack):
        """
        Return (image width, image height).
        """
        first_sec, last_sec = DataManager.load_cropbox(stack)[4:]
        anchor_fn = DataManager.load_anchor_filename(stack)
        filename_to_section, section_to_filename = DataManager.load_sorted_filenames(stack)
        while True:
            random_fn = section_to_filename[np.random.randint(first_sec, last_sec+1, 1)[0]]
            fn = DataManager.get_image_filepath(stack=stack, resol='lossless', version='compressed', data_dir=data_dir, fn=random_fn, anchor_fn=anchor_fn)
            if not os.path.exists(fn):
                fn = DataManager.get_image_filepath(stack=stack, resol='lossless', version='saturation', data_dir=data_dir, fn=random_fn, anchor_fn=anchor_fn)
                if not os.path.exists(fn):
                    continue
            image_width, image_height = map(int, check_output("identify -format %%Wx%%H %s" % fn, shell=True).split('x'))
            break

        return image_width, image_height

    @staticmethod
    def convert_section_to_z(stack, sec, downsample, z_begin=None, first_sec=None):
        """
        first_sec: default to the first brainstem section defined in ``cropbox".
        z_begin: default to the z position of the first_sec.
        """

        xy_pixel_distance = XY_PIXEL_DISTANCE_LOSSLESS * downsample
        voxel_z_size = SECTION_THICKNESS / xy_pixel_distance
        # print 'voxel size:', xy_pixel_distance, xy_pixel_distance, voxel_z_size, 'um'

        # first_sec, last_sec = section_range_lookup[stack]
        if first_sec is None:
            first_sec, _ = DataManager.load_cropbox(stack)[4:]

        # z_end = int(np.ceil((last_sec+1)*voxel_z_size))
        if z_begin is None:
            # z_begin = int(np.floor(first_sec*voxel_z_size))
            z_begin = first_sec * voxel_z_size
        # print 'z_begin', first_sec*voxel_z_size, z_begin

        z1 = sec * voxel_z_size
        z2 = (sec + 1) * voxel_z_size
        # return int(z1)-z_begin, int(z2)+1-z_begin
        # print 'z1, z2', z1-z_begin, z2-1-z_begin
        return z1-z_begin, z2-1-z_begin
        # return int(np.round(z1-z_begin)), int(np.round(z2-1-z_begin))
        # return int(np.round(z1))-z_begin, int(np.round(z2))-1-z_begin

    @staticmethod
    def convert_z_to_section(stack, z, downsample, z_begin=None):
        """
        z_begin default to int(np.floor(first_sec*voxel_z_size)).
        """
        xy_pixel_distance = XY_PIXEL_DISTANCE_LOSSLESS * downsample
        voxel_z_size = SECTION_THICKNESS / xy_pixel_distance
        # print 'voxel size:', xy_pixel_distance, xy_pixel_distance, voxel_z_size, 'um'

        # first_sec, last_sec = section_range_lookup[stack]
        first_sec, last_sec = DataManager.load_cropbox(stack)[4:]
        # z_end = int(np.ceil((last_sec+1)*voxel_z_size))

        if z_begin is None:
            # z_begin = int(np.floor(first_sec*voxel_z_size))
            z_begin = first_sec * voxel_z_size
        # print 'z_begin', first_sec*voxel_z_size, z_begin

        sec_float = np.float32((z + z_begin) / voxel_z_size) # if use np.float, will result in np.floor(98.0)=97
        # print sec_float
        # print sec_float == 98., np.floor(np.float(sec_float))
        sec_floor = int(np.floor(sec_float))

        return sec_floor


    @staticmethod
    def get_initial_snake_contours_filepath(stack):
        """"""
        anchor_fn = metadata_cache['anchor_fn'][stack]
        init_snake_contours_fp = os.path.join(THUMBNAIL_DATA_DIR, stack, stack+'_alignedTo_'+anchor_fn+'_init_snake_contours.pkl')
        return init_snake_contours_fp

    @staticmethod
    def get_auto_submask_rootdir_filepath(stack):
        """
        Args:
            what (str): submask or decisions.
            submask_ind (int): if what is submask, must provide submask_ind.
        """
        anchor_fn = metadata_cache['anchor_fn'][stack]
        dir_path = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_auto_submasks')
        return dir_path

    @staticmethod
    def get_auto_submask_dir_filepath(stack, fn=None, sec=None):
        """
        Args:
            what (str): submask or decisions.
            submask_ind (int): if what is submask, must provide submask_ind.
        """
        auto_submasks_dir = DataManager.get_auto_submask_rootdir_filepath(stack)
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        dir_path = os.path.join(auto_submasks_dir, fn)
        return dir_path

    @staticmethod
    def get_auto_submask_filepath(stack, what, submask_ind=None, fn=None, sec=None):
        """
        Args:
            what (str): submask or decisions.
            submask_ind (int): if what is submask, must provide submask_ind.
        """
        anchor_fn = metadata_cache['anchor_fn'][stack]
        dir_path = DataManager.get_auto_submask_dir_filepath(stack=stack, fn=fn, sec=sec)

        if what == 'submask':
            assert submask_ind is not None, "Must provide submask_ind."
            fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_auto_submask_%d.png' % submask_ind)
        elif what == 'decisions':
            fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_auto_submask_decisions.csv')
        else:
            raise Exception("Not recognized.")

        return fp

    @staticmethod
    def get_user_modified_submask_rootdir_filepath(stack):
        """
        Args:
            what (str): submask or decisions.
            submask_ind (int): if what is submask, must provide submask_ind.
        """
        anchor_fn = metadata_cache['anchor_fn'][stack]
        dir_path = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_userModified_submasks')
        return dir_path

    @staticmethod
    def get_user_modified_submask_dir_filepath(stack, fn=None, sec=None):
        """
        Args:
            what (str): submask or decisions.
            submask_ind (int): if what is submask, must provide submask_ind.
        """
        auto_submasks_dir = DataManager.get_user_modified_submask_rootdir_filepath(stack)
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        dir_path = os.path.join(auto_submasks_dir, fn)
        return dir_path

    @staticmethod
    def get_user_modified_submask_filepath(stack, what, submask_ind=None, fn=None, sec=None):
        """
        Args:
            what (str): submask or decisions.
            submask_ind (int): if what is submask, must provide submask_ind.
        """
        anchor_fn = metadata_cache['anchor_fn'][stack]
        dir_path = DataManager.get_user_modified_submask_dir_filepath(stack=stack, fn=fn, sec=sec)

        if what == 'submask':
            assert submask_ind is not None, "Must provide submask_ind."
            fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_userModified_submask_%d.png' % submask_ind)
        elif what == 'decisions':
            fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_userModified_submask_decisions.csv')
        elif what == 'parameters':
            fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_userModified_parameters.json')
        elif what == 'contour_vertices':
            fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_userModified_submask_contour_vertices.pkl')
        else:
            raise Exception("Not recognized.")

        return fp

    @staticmethod
    def get_mask_dirpath(stack, version='aligned'):
        """
        Get directory path of thumbnail mask.
        
        Args:
            version (str): One of aligned, aligned_cropped, cropped.
        """
        
        anchor_fn = metadata_cache['anchor_fn'][stack]
        if version == 'aligned':
            dir_path = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_masks')
        elif version == 'aligned_cropped' or version == 'cropped':
            dir_path = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_alignedTo_' + anchor_fn + '_masks_cropped')
        else:
            raise Exception('version %s is not recognized.' % version)
        return dir_path

    @staticmethod
    def get_mask_filepath(stack, sec=None, fn=None, version='aligned'):
        """
        Get filepath of thumbnail mask.
        
        Args:
            version (str): One of aligned, aligned_cropped, cropped.
        """
        
        anchor_fn = metadata_cache['anchor_fn'][stack]
        dir_path = DataManager.get_mask_dirpath(stack, version=version)
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]

        if version == 'aligned':
            fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_mask.png')
        elif version == 'aligned_cropped' or version == 'cropped':
            fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_mask_cropped.png')
        else:
            raise Exception('version %s is not recognized.' % version)
        return fp
    
    @staticmethod
    def get_region_labels_filepath(stack, sec=None, fn=None):
        """
        Returns:
            dict {label: list of region indices}   
        """
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        return os.path.join(CELL_FEATURES_CLF_ROOTDIR, 'region_indices_by_label', stack, fn + '_region_indices_by_label.hdf')
        

##################################################

# This module stores any meta information that is dynamic.
metadata_cache = {}

def generate_metadata_cache():

    global metadata_cache
    # metadata_cache['image_shape'] = {stack: DataManager.get_image_dimension(stack) for stack in all_stacks}
    metadata_cache['image_shape'] =\
    {'MD585': (16384, 12000),
     'MD589': (15520, 11936),
     'MD590': (17536, 13056),
     'MD591': (16000, 13120),
     'MD592': (17440, 12384),
     'MD593': (17088, 12256),
     'MD594': (17216, 11104),
     'MD595': (18368, 13248),
     'MD598': (18400, 12608),
     'MD599': (18784, 12256),
     'MD602': (22336, 12288),
     'MD603': (20928, 13472),
     'MD635': (20960, 14240),
     'MD642': (28704, 15584),
     'MD657': (27584, 16960),
     'MD658': (19936, 15744)}
    metadata_cache['anchor_fn'] = {}
    metadata_cache['sections_to_filenames'] = {}
    metadata_cache['section_limits'] = {}
    metadata_cache['cropbox'] = {}
    for stack in all_stacks:
        try:
            metadata_cache['anchor_fn'][stack] = DataManager.load_anchor_filename(stack)
        except:
            pass
        try:
            metadata_cache['sections_to_filenames'][stack] = DataManager.load_sorted_filenames(stack)[1]
        except:
            pass
        try:
            metadata_cache['section_limits'][stack] = DataManager.load_cropbox(stack)[4:]
        except:
            pass
        try:
            metadata_cache['cropbox'][stack] = DataManager.load_cropbox(stack)[:4]
        except:
            pass

generate_metadata_cache()


# # metadata_cache['image_shape'] = {stack: DataManager.get_image_dimension(stack) for stack in all_stacks}
# metadata_cache['image_shape'] =\
# {'MD585': (16384, 12000),
#  'MD589': (15520, 11936),
#  'MD590': (17536, 13056),
#  'MD591': (16000, 13120),
#  'MD592': (17440, 12384),
#  'MD593': (17088, 12256),
#  'MD594': (17216, 11104),
#  'MD595': (18368, 13248),
#  'MD598': (18400, 12608),
#  'MD599': (18784, 12256),
#  'MD602': (22336, 12288),
#  'MD603': (20928, 13472),
#  'MD635': (20960, 14240),
#  'MD642': (28704, 15584),
#  'MD657': (27584, 16960)}
# metadata_cache['anchor_fn'] = {}
# metadata_cache['sections_to_filenames'] = {}
# metadata_cache['section_limits'] = {}
# metadata_cache['cropbox'] = {}
# for stack in all_stacks:
#     try:
#         metadata_cache['anchor_fn'][stack] = DataManager.load_anchor_filename(stack)
#     except:
#         pass
#     try:
#         metadata_cache['sections_to_filenames'][stack] = DataManager.load_sorted_filenames(stack)[1]
#     except:
#         pass
#     try:
#         metadata_cache['section_limits'][stack] = DataManager.load_cropbox(stack)[4:]
#     except:
#         pass
#     try:
#         metadata_cache['cropbox'][stack] = DataManager.load_cropbox(stack)[:4]
#     except:
#         pass

# del stack

def resolve_actual_setting(setting, stack, fn=None, sec=None):
    """Take a possibly composite setting index, and return the actual setting index according to fn."""

    if stack in all_nissl_stacks:
        stain = 'nissl'
    elif stack in all_ntb_stacks:
        stain = 'ntb'
    elif stack in all_alt_nissl_ntb_stacks:
        if fn is None:
            assert sec is not None
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        if fn.split('-')[1][0] == 'F':
            stain = 'ntb'
        elif fn.split('-')[1][0] == 'N':
            stain = 'nissl'
        else:
            raise Exception('Must be either ntb or nissl.')

    if setting == 12:
        setting_nissl = 2
        setting_ntb = 10

    if setting == 12:
        if stain == 'nissl':
            setting_ = setting_nissl
        else:
            setting_ = setting_ntb
    else:
        setting_ = setting

    return setting_
