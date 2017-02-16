import sys, os
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from vis3d_utilities import *

from pandas import read_hdf

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

    @staticmethod
    def map_local_filename_to_s3(local_fp):
        # Saienthan, please implement this.
        return None

    @staticmethod
    def load_data(filepath, filetype):

        if not os.path.exists(filepath):
            sys.stderr.write('File does not exist: %s\n' % filepath)

            # If on aws, download from S3 and make available locally.
            on_aws = False # should be an env variable
            if on_aws:
                DataManager.download_from_s3(filepath, DataManager.map_local_filename_to_s3(filepath))

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

    @staticmethod
    def get_volume_label_to_name_filename(stack):
        fn = os.path.join(volume_dir, stack, stack+'_down32_annotationVolume_nameToLabel.txt')
        return fn

    @staticmethod
    def load_volume_label_to_name(stack):
        fn = DataManager.get_volume_label_to_name_filename(stack)
        label_to_name, name_to_label = DataManager.load_data(fn, filetype='label_name_map')
        return label_to_name, name_to_label

    # @staticmethod
    # def load_volume_bbox(stack):
    #     with open(os.path.join(volume_dir, stack, stack+'_down32_annotationVolume_bbox.txt'), 'r') as f:
    #         bbox = map(int, f.readline().strip().split())
    #     return bbox

    @staticmethod
    def get_anchor_filename_filename(stack):
        fn = thumbnail_data_dir + '/%(stack)s/%(stack)s_anchor.txt' % dict(stack=stack)
        return fn

    @staticmethod
    def load_anchor_filename(stack):
        fn = DataManager.get_anchor_filename_filename(stack)
        anchor_fn = DataManager.load_data(fn, filetype='anchor')
        return anchor_fn

    @staticmethod
    def get_cropbox_filename(stack):
        fn = thumbnail_data_dir + '/%(stack)s/%(stack)s_cropbox.txt' % dict(stack=stack)
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
        fn = thumbnail_data_dir + '/%(stack)s/%(stack)s_sorted_filenames.txt' % dict(stack=stack)
        return fn

    @staticmethod
    def load_sorted_filenames(stack):
        fn = DataManager.get_sorted_filenames_filename(stack)
        filename_to_section, section_to_filename = DataManager.load_data(fn, filetype='file_section_map')
        return filename_to_section, section_to_filename

    @staticmethod
    def get_transforms_filename(stack):
        fn = thumbnail_data_dir + '/%(stack)s/%(stack)s_elastix_output/%(stack)s_transformsTo_anchor.pkl' % dict(stack=stack)
        return fn

    @staticmethod
    def load_transforms(stack, downsample_factor):

        fn = DataManager.get_transforms_filename(stack)
        Ts = DataManager.load_data(fn, filetype='pickle')

        Ts_inv_downsampled = {}
        for fn, T0 in Ts.iteritems():
            T = T0.copy()
            T[:2, 2] = T[:2, 2] * 32 / downsample_factor
            Tinv = np.linalg.inv(T)
            Ts_inv_downsampled[fn] = Tinv

        return Ts_inv_downsampled

    # @staticmethod
    # def save_thumbnail_mask(mask, stack, section, cerebellum_removed=False):
    #
    #     fn = DataManager.get_thumbnail_mask_filepath(stack, section, cerebellum_removed=cerebellum_removed)
    #     create_if_not_exists(os.path.dirname(fn))
    #     imsave(fn, mask)
    #     sys.stderr.write('Thumbnail mask for section %s, %d saved to %s.\n' % (stack, section, fn))

    # @staticmethod
    # def load_thumbnail_mask(stack, section, cerebellum_removed=False):
    #     fn = DataManager.get_thumbnail_mask_filepath(stack, section, cerebellum_removed=cerebellum_removed)
    #     thumbmail_mask = DataManager.load_data(fn, filetype='image').astype(np.bool)
    #     return thumbmail_mask

    @staticmethod
    def get_thumbnail_mask_filename_v2(stack, section=None, version='aligned_cropped'):
        # anchor_fn = DataManager.load_anchor_filename(stack)
        anchor_fn = metadata_cache['anchor_fn'][stack]
        # filename_to_section, section_to_filename = DataManager.load_sorted_filenames(stack)
        sections_to_filenames = metadata_cache['sections_to_filenames'][stack]
        fn = sections_to_filenames[section]

        if version == 'aligned_cropped':
            #
            # image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_unsorted_alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn})
            # image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn}])
            # image_path = os.path.join(image_dir, image_name + '.jpg')

            fn = thumbnail_data_dir+'/%(stack)s/%(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s_cropped/%(fn)s_mask_alignedTo_%(anchor_fn)s_cropped.png' % \
                dict(stack=stack, fn=fn, anchor_fn=anchor_fn)
        elif version == 'aligned':
            fn = thumbnail_data_dir+'/%(stack)s/%(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s/%(stack)s_%(sec)04d_mask_alignedTo_%(anchor_fn)s.png' % \
                dict(stack=stack, fn=fn, anchor_fn=anchor_fn)

        # if version == 'aligned_cropped':
        #     fn = data_dir+'/%(stack)s/%(stack)s_mask_unsorted_aligned_cropped/%(stack)s_%(sec)04d_mask_aligned_cropped.png' % \
        #         dict(stack=stack, sec=section)
        # elif version == 'aligned':
        #     fn = data_dir+'/%(stack)s/%(stack)s_mask_sorted_aligned/%(stack)s_%(sec)04d_mask_aligned.png' % \
        #         dict(stack=stack, sec=section)
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
    def get_local_alignment_viz_dir(stack_fixed, stack_moving, fixed_volume_type='score', moving_volume_type='score', train_sample_scheme=None, global_transform_scheme=None, local_transform_scheme=None):

        atlasAlignParams_dir = atlasAlignParams_rootdir + '/%(stack_moving)s_over_%(stack_fixed)s' % \
                     {'stack_moving': stack_moving, 'stack_fixed': stack_fixed}

        viz_dir = atlasAlignParams_dir + '/%(stack_moving)s_down32_%(m_str)s_over_%(stack_fixed)s_down32_%(f_str)s_trainSampleScheme_%(scheme)d_globalTxScheme_%(gtf_sheme)d_localTxScheme_%(ltf_sheme)d_viz' % \
                    {'stack_moving': stack_moving, 'stack_fixed': stack_fixed,
                    'm_str': volume_type_to_str(moving_volume_type), 'f_str': volume_type_to_str(fixed_volume_type),
                    'scheme':train_sample_scheme, 'gtf_sheme':global_transform_scheme,
                    'ltf_sheme':local_transform_scheme}
        return viz_dir


    @staticmethod
    def get_local_alignment_parameters_filepath_prefix(stack_fixed, fixed_volume_type, stack_moving, moving_volume_type='score', label=None,
    train_sample_scheme=1, global_transform_scheme=1, local_transform_scheme=1):

        atlasAlignParams_dir = atlasAlignParams_rootdir + '/%(stack_moving)s_over_%(stack_fixed)s' % \
                     {'stack_moving': stack_moving, 'stack_fixed': stack_fixed}

        return atlasAlignParams_dir + '/%(stack_moving)s_down32_%(m_str)s_over_%(stack_fixed)s_down32_%(f_str)s_%(label)s_trainSampleScheme_%(scheme)d_globalTxScheme_%(gtf_sheme)d_localTxScheme_%(ltf_sheme)d' % \
                  {'stack_moving': stack_moving, 'stack_fixed': stack_fixed,
                  'label': label,
                  'm_str': volume_type_to_str(moving_volume_type), 'f_str': volume_type_to_str(fixed_volume_type),
                  'scheme':train_sample_scheme, 'gtf_sheme':global_transform_scheme, 'ltf_sheme':local_transform_scheme}

    @staticmethod
    def get_local_alignment_parameters_filepath(stack_fixed, stack_moving, fixed_volume_type='score', moving_volume_type='score', label=None,
    train_sample_scheme=1, global_transform_scheme=1, local_transform_scheme=1, trial_idx=None):
        partial_fn = DataManager.get_local_alignment_parameters_filepath_prefix(stack_fixed=stack_fixed,
        fixed_volume_type=fixed_volume_type, stack_moving=stack_moving, moving_volume_type=moving_volume_type,
        train_sample_scheme=train_sample_scheme, global_transform_scheme=global_transform_scheme, local_transform_scheme=local_transform_scheme,
        label=label)

        if trial_idx is not None:
            return partial_fn + '_parameters_trial_%d.txt' % trial_idx
        else:
            return partial_fn + '_parameters.txt'

    @staticmethod
    def load_local_alignment_parameters(stack_fixed, stack_moving, fixed_volume_type='score', moving_volume_type='score', label=None,
    train_sample_scheme=1, global_transform_scheme=1, local_transform_scheme=1, trial_idx=None):

        params_fp = DataManager.get_local_alignment_parameters_filepath(stack_fixed=stack_fixed,
                    fixed_volume_type=fixed_volume_type, stack_moving=stack_moving, moving_volume_type=moving_volume_type,
                    train_sample_scheme=train_sample_scheme, global_transform_scheme=global_transform_scheme, local_transform_scheme=local_transform_scheme,
                    label=label,
                    trial_idx=trial_idx)

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
    def get_local_alignment_score_plot_filepath(stack_fixed, stack_moving, fixed_volume_type='score', moving_volume_type='score', label=None,
    train_sample_scheme=1, global_transform_scheme=1, local_transform_scheme=1, trial_idx=None):
        partial_fn = DataManager.get_local_alignment_parameters_filepath_prefix(stack_fixed=stack_fixed,
        fixed_volume_type=fixed_volume_type, stack_moving=stack_moving, moving_volume_type=moving_volume_type,
        train_sample_scheme=train_sample_scheme, global_transform_scheme=global_transform_scheme, local_transform_scheme=local_transform_scheme,
        label=label)

        if trial_idx is not None:
            return partial_fn + '_scoreEvolution_trial_%d.png' % trial_idx
        else:
            return partial_fn + '_scoreEvolution.png'


    @staticmethod
    def get_global_alignment_parameters_filepath_prefix(stack_fixed, stack_moving, fixed_volume_type='score', moving_volume_type='score', train_sample_scheme=None, global_transform_scheme=None):

        atlasAlignParams_dir = atlasAlignParams_rootdir + '/%(stack_moving)s_to_%(stack_fixed)s' % \
                     {'stack_moving': stack_moving, 'stack_fixed': stack_fixed}

        suffix = generate_suffix(train_sample_scheme=train_sample_scheme, global_transform_scheme=global_transform_scheme)

        if suffix == '':
            return atlasAlignParams_dir + '/%(stack_moving)s_down32_%(m_str)s_to_%(stack_fixed)s_down32_%(f_str)s' % \
                      {'stack_moving': stack_moving, 'stack_fixed': stack_fixed,
                      'm_str': volume_type_to_str(moving_volume_type), 'f_str': volume_type_to_str(fixed_volume_type)}
        else:
            return atlasAlignParams_dir + '/%(stack_moving)s_down32_%(m_str)s_to_%(stack_fixed)s_down32_%(f_str)s_%(suffix)s' % \
                      {'stack_moving': stack_moving, 'stack_fixed': stack_fixed,
                      'm_str': volume_type_to_str(moving_volume_type), 'f_str': volume_type_to_str(fixed_volume_type),
                      'suffix': suffix}

    @staticmethod
    def get_global_alignment_parameters_filepath(stack_fixed, stack_moving, fixed_volume_type='score', moving_volume_type='score', train_sample_scheme=None, global_transform_scheme=None, trial_idx=None):
        partial_fn = DataManager.get_global_alignment_parameters_filepath_prefix(stack_fixed, stack_moving, fixed_volume_type, moving_volume_type, train_sample_scheme, global_transform_scheme)

        if trial_idx is None:
            return partial_fn + '_parameters.txt'
        else:
            return partial_fn + '_parameters_trial_%d.txt' % trial_idx

    @staticmethod
    def load_global_alignment_parameters(stack_fixed, stack_moving, fixed_volume_type='score', moving_volume_type='score', train_sample_scheme=None, global_transform_scheme=None, trial_idx=None):

        params_fp = DataManager.get_global_alignment_parameters_filepath(stack_moving=stack_moving,
                                                                    moving_volume_type=moving_volume_type,
                                                                    stack_fixed=stack_fixed,
                                                                    fixed_volume_type=fixed_volume_type,
                                                                    train_sample_scheme=train_sample_scheme,
                                                                    global_transform_scheme=global_transform_scheme,
                                                                    trial_idx=trial_idx)
        return DataManager.load_data(params_fp, 'transform_params')


    @staticmethod
    def get_global_alignment_score_plot_filepath(stack_fixed, stack_moving, fixed_volume_type='score', moving_volume_type='score', train_sample_scheme=None, global_transform_scheme=None, trial_idx=None):
        partial_fn = DataManager.get_global_alignment_parameters_filepath_prefix(stack_fixed, stack_moving, fixed_volume_type, moving_volume_type, train_sample_scheme, global_transform_scheme)

        if trial_idx is None:
            return partial_fn + '_scoreEvolution.png'
        else:
            return partial_fn + '_scoreEvolution_trial_%d.png' % trial_idx

    @staticmethod
    def get_global_alignment_viz_dir(stack_fixed, stack_moving,
    fixed_volume_type='score', moving_volume_type='score',
    train_sample_scheme=None, global_transform_scheme=None):

        # clf_suffix = generate_suffix(train_sample_scheme=train_sample_scheme)
        # gtf_suffix = generate_suffix(global_transform_scheme=global_transform_scheme)

        suffix = generate_suffix(train_sample_scheme=train_sample_scheme,
                                    global_transform_scheme=global_transform_scheme)

        atlasAlignParams_dir = atlasAlignParams_rootdir + '/%(stack_moving)s_to_%(stack_fixed)s' % \
                     {'stack_moving': stack_moving, 'stack_fixed': stack_fixed}

        viz_dir = atlasAlignParams_dir + '/%(stack_moving)s_down32_%(m_str)s_to_%(stack_fixed)s_down32_%(f_str)s%(suffix)s_viz' % \
                    {'stack_moving': stack_moving, 'stack_fixed': stack_fixed,
                    'm_str': volume_type_to_str(moving_volume_type), 'f_str': volume_type_to_str(fixed_volume_type),
                    'suffix': '_' + suffix if suffix != '' else ''
                    }

        return viz_dir

    @staticmethod
    def get_zscore_filepath(stack_moving, stack_fixed, moving_volume_type, fixed_volume_type,
                            train_sample_scheme, global_transform_scheme, local_transform_scheme=None, label=None):
        if label is not None:
            # local
            return os.path.join(HESSIAN_ROOTDIR,
                    '%(stack_moving)s_to_%(stack_fixed)s' % \
                    {'stack_moving': stack_moving, 'stack_fixed': stack_fixed},
                    '%(stack_moving)s_down32_%(m_str)s_to_%(stack_fixed)s_down32_%(f_str)s_%(label)s_trainSampleScheme_%(scheme)d_globalTxScheme_%(gtf_sheme)d_localTxScheme_%(ltf_sheme)d_zscores.pkl' % \
                                {'stack_moving': stack_moving, 'stack_fixed': stack_fixed,
                                'm_str': volume_type_to_str(moving_volume_type), 'f_str': volume_type_to_str(fixed_volume_type),
                                'scheme':train_sample_scheme, 'gtf_sheme':global_transform_scheme, 'ltf_sheme': local_transform_scheme,
                                'label': label}
                                )
        else:
            # global
            assert local_transform_scheme is None
            return os.path.join(HESSIAN_ROOTDIR,
                    '%(stack_moving)s_to_%(stack_fixed)s' % \
                    {'stack_moving': stack_moving, 'stack_fixed': stack_fixed},
                    '%(stack_moving)s_down32_%(m_str)s_to_%(stack_fixed)s_down32_%(f_str)s_trainSampleScheme_%(scheme)d_globalTxScheme_%(gtf_sheme)d_zscores.pkl' % \
                                {'stack_moving': stack_moving, 'stack_fixed': stack_fixed,
                                'm_str': volume_type_to_str(moving_volume_type), 'f_str': volume_type_to_str(fixed_volume_type),
                                'scheme':train_sample_scheme, 'gtf_sheme':global_transform_scheme}
                                )

    @staticmethod
    def get_hessian_filepath(stack_moving, stack_fixed, moving_volume_type, fixed_volume_type,
                            train_sample_scheme, global_transform_scheme, local_transform_scheme=None,
                             label=None):
        if label is not None:
            # local
            return os.path.join(HESSIAN_ROOTDIR,
                    '%(stack_moving)s_to_%(stack_fixed)s' % \
                    {'stack_moving': stack_moving, 'stack_fixed': stack_fixed},
                    '%(stack_moving)s_down32_%(m_str)s_to_%(stack_fixed)s_down32_%(f_str)s_%(label)s_trainSampleScheme_%(scheme)d_globalTxScheme_%(gtf_sheme)d_localTxScheme_%(ltf_sheme)d_hessians.pkl' % \
                                {'stack_moving': stack_moving, 'stack_fixed': stack_fixed,
                                'm_str': volume_type_to_str(moving_volume_type), 'f_str': volume_type_to_str(fixed_volume_type),
                                'scheme':train_sample_scheme, 'gtf_sheme':global_transform_scheme, 'ltf_sheme': local_transform_scheme,
                                'label': label}
                                )
        else:
            # global
            assert local_transform_scheme is None
            return os.path.join(HESSIAN_ROOTDIR,
                    '%(stack_moving)s_to_%(stack_fixed)s' % \
                    {'stack_moving': stack_moving, 'stack_fixed': stack_fixed},
                    '%(stack_moving)s_down32_%(m_str)s_to_%(stack_fixed)s_down32_%(f_str)s_trainSampleScheme_%(scheme)d_globalTxScheme_%(gtf_sheme)d_hessians.pkl' % \
                                {'stack_moving': stack_moving, 'stack_fixed': stack_fixed,
                                'm_str': volume_type_to_str(moving_volume_type), 'f_str': volume_type_to_str(fixed_volume_type),
                                'scheme':train_sample_scheme, 'gtf_sheme':global_transform_scheme}
                                )

    # @staticmethod
    # def get_classifier_filepath(label, train_sample_scheme=None):
    #     return CLF_ROOTDIR + '/classifiers/%(label)s_clf_%(suffix)s.dump' % \
    #     {'label': label, 'suffix':'trainSampleScheme_%d' % train_sample_scheme}
    #
    #
    # @staticmethod
    # def get_classifier_neurotraceBlue_filepath(label, train_sample_scheme=None):
    #     return CLF_NTBLUE_ROOTDIR + '/classifiers/%(label)s_clf_%(suffix)s.dump' % \
    #     {'label': label, 'suffix':'trainSampleScheme_%d' % train_sample_scheme}

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

        return DataManager.load_data(sparse_scores_fn, filetype='bp')


    # @staticmethod
    # def load_sparse_scores(stack, sec=None, fn=None, anchor_fn=None, label='', train_sample_scheme=None):
    #
    #     if fn is None:
    #         fn = metadata_cache['sections_to_filenames'][stack][sec]
    #
    #     if anchor_fn is None:
    #         anchor_fn = metadata_cache['anchor_fn'][stack]
    #
    #     sparse_scores_fn = DataManager.get_sparse_scores_filepath(stack=stack, fn=fn, anchor_fn=anchor_fn,
    #         label=label, train_sample_scheme=train_sample_scheme)
    #
    #     return DataManager.load_data(sparse_scores_fn, filetype='bp')

    @staticmethod
    def get_sparse_scores_filepath(stack, structure, setting, sec=None, fn=None, anchor_fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        # setting_suffix = []
        # if setting_ntb is not None:
        #     setting_suffix.append('settingNtb_' + str(setting_ntb))
        # if setting_nissl is not None:
        #     setting_suffix.append('settingNissl_' + str(setting_nissl))
        # setting_suffix = '_'.join(suffix)

        return os.path.join(SPARSE_SCORES_ROOTDIR, stack, '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped', \
                '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_sparseScores_setting_%(setting)s.hdf') % \
                {'fn': fn, 'anchor_fn': anchor_fn, 'structure':structure, 'setting': setting}

    # @staticmethod
    # def get_sparse_scores_filepath(stack, sec=None, fn=None, anchor_fn=None, label=None, train_sample_scheme=None):
    #     if fn is None:
    #         fn = metadata_cache['sections_to_filenames'][stack][sec]
    #
    #     if anchor_fn is None:
    #         anchor_fn = metadata_cache['anchor_fn'][stack]
    #
    #     suffix = generate_suffix(train_sample_scheme=train_sample_scheme)
    #     return os.path.join(SPARSE_SCORES_ROOTDIR, stack, '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped', \
    #             '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(label)s_sparseScores%(suffix)s.hdf') % \
    #             {'fn': fn, 'anchor_fn': anchor_fn, 'label':label, 'suffix': '_' + suffix if suffix != '' else ''}

    @staticmethod
    def load_annotation_volume(stack, downscale):
        fn = DataManager.get_annotation_volume_filepath(stack, downscale)
        return DataManager.load_data(fn, filetype='bp')

    @staticmethod
    def get_annotation_volume_filepath(stack, downscale):
        vol_fn = VOLUME_ROOTDIR + '/%(stack)s/%(stack)s_down%(ds)d_annotationVolume.bp' % \
                {'stack':stack, 'ds':downscale}
        return vol_fn


    @staticmethod
    def get_annotation_volume_bbox_filepath(stack, downscale):
        vol_fn = VOLUME_ROOTDIR + '/%(stack)s/%(stack)s_down%(ds)d_annotationVolume_bbox.txt' % \
                {'stack':stack, 'ds':downscale}
        return vol_fn

    @staticmethod
    def get_annotation_volume_nameToLabel_filepath(stack, downscale):
        vol_fn = VOLUME_ROOTDIR + '/%(stack)s/%(stack)s_down%(ds)d_annotationVolume_nameToLabel.txt' % \
                {'stack':stack, 'ds':downscale}
        return vol_fn

    @staticmethod
    def load_annotation_volume_nameToLabel(stack, downscale):
        fn = DataManager.get_annotation_volume_nameToLabel_filepath(stack, downscale)

        labels_to_names, names_to_labels = DataManager.load_data(fn, filetype='label_name_map')

        # name_to_label = {}
        # with open(fn, 'r') as f:
        #     for line in f.readlines():
        #         name, label_str = line.split()
        #         name_to_label[name] =  int(label_str)

        return names_to_labels


    @staticmethod
    def load_transformed_volume(stack_m, type_m, stack_f, type_f, downscale,
                                        train_sample_scheme_m=None, train_sample_scheme_f=None,
                                        global_transform_scheme=None,
                                        local_transform_scheme=None,
                                        label=None,
                                        transitive=None):
        fp = DataManager.get_transformed_volume_filepath(stack_m=stack_m, type_m=type_m, stack_f=stack_f, type_f=type_f,
                                        downscale=downscale,
                                            train_sample_scheme_m=train_sample_scheme_m,
                                            train_sample_scheme_f=train_sample_scheme_f,
                                            global_transform_scheme=global_transform_scheme,
                                            local_transform_scheme=local_transform_scheme,
                                            label=label,
                                            transitive=transitive)
        return DataManager.load_data(fp, filetype='bp')


    @staticmethod
    def get_transformed_volume_file_basename(stack_m, type_m, stack_f, type_f, downscale,
                                        train_sample_scheme_m=None, train_sample_scheme_f=None,
                                        global_transform_scheme=None,
                                        local_transform_scheme=None,
                                        label=None,
                                        transitive=None):
        clf_suffix_m = generate_suffix(train_sample_scheme=train_sample_scheme_m)
        clf_suffix_f = generate_suffix(train_sample_scheme=train_sample_scheme_f)
        gtf_suffix = generate_suffix(global_transform_scheme=global_transform_scheme)
        ltf_suffix = generate_suffix(local_transform_scheme=local_transform_scheme)

        def add_underscore(x):
            return '' if x == '' or x is None else '_' + x

        if transitive is None:
            if local_transform_scheme is None:
                # loading globally transformed volume
                transitive = 'to'
            else:
                # loading locally transformed volume
                transitive = 'over'

        assert transitive in ['to', 'over', 'by'], 'transitive must be among [to, over, by]'

        if transitive == 'by':
            vol_fn_basename = '%(stack_m)s_over_%(stack_f)s_to_%(stack_m)s/%(stack_m)s_down%(ds)d_%(type_m_str)s%(clf_suffix_m)s_over_%(stack_f)s_down%(ds)d_%(type_f_str)s_to_%(stack_m)s%(label)s%(clf_suffix_f)s%(gtf_suffix)s%(ltf_suffix)s' % \
                    {'stack_m':stack_m,
                    'stack_f':stack_f,
                    'type_m_str': volume_type_to_str(type_m),
                    'type_f_str': volume_type_to_str(type_f),
                    'ds':downscale,
                    'transitive': transitive,
                    'label': add_underscore(label),
                    'clf_suffix_m': '_' + clf_suffix_m if clf_suffix_m != '' else '',
                    'clf_suffix_f': '_' + clf_suffix_f if clf_suffix_f != '' else '',
                    'gtf_suffix': '_' + gtf_suffix if gtf_suffix != '' else '',
                    'ltf_suffix': '_' + ltf_suffix if ltf_suffix != '' else ''}
        else:
            vol_fn_basename = '%(stack_m)s_%(transitive)s_%(stack_f)s/%(stack_m)s_down%(ds)d_%(type_m_str)s%(clf_suffix_m)s_%(transitive)s_%(stack_f)s_down%(ds)d_%(type_f_str)s%(label)s%(clf_suffix_f)s%(gtf_suffix)s%(ltf_suffix)s' % \
                    {'stack_m':stack_m,
                    'stack_f':stack_f,
                    'type_m_str': volume_type_to_str(type_m),
                    'type_f_str': volume_type_to_str(type_f),
                    'ds':downscale,
                    'transitive': transitive,
                    'label': add_underscore(label),
                    'clf_suffix_m': '_' + clf_suffix_m if clf_suffix_m != '' else '',
                    'clf_suffix_f': '_' + clf_suffix_f if clf_suffix_f != '' else '',
                    'gtf_suffix': '_' + gtf_suffix if gtf_suffix != '' else '',
                    'ltf_suffix': '_' + ltf_suffix if ltf_suffix != '' else ''}

        return vol_fn_basename


    ###################################
    # Mesh related
    ###################################

    @staticmethod
    def load_shell_mesh(stack, downscale, return_polydata_only=True):
        shell_mesh_fn = DataManager.get_shell_mesh_filepath(stack, downscale)
        return load_mesh_stl(shell_mesh_fn, return_polydata_only=return_polydata_only)

    @staticmethod
    def get_shell_mesh_filepath(stack, downscale):
        shell_mesh_fn = os.path.join(MESH_ROOTDIR, stack, "%(stack)s_down%(ds)d_outerContourVolume_smoothed.stl" % {'stack': stack, 'ds':downscale})
        return shell_mesh_fn

    @staticmethod
    def load_meshes(stack, labels, return_polydata_only=True):
        meshes = {label: load_mesh_stl(DataManager.get_mesh_filepath(stack, label), return_polydata_only=return_polydata_only)
         for label in labels}
        return meshes

    @staticmethod
    def load_mesh(stack, label, return_polydata_only=True):
        mesh_fn = DataManager.get_mesh_filepath(stack, label)
        return load_mesh_stl(mesh_fn, return_polydata_only=return_polydata_only)

    @staticmethod
    def get_mesh_filepath(stack, label):
        mesh_fn = os.path.join(MESH_ROOTDIR, stack, 'structure_mesh', 'mesh_%s.stl'%label)
        return mesh_fn

    @staticmethod
    def get_annotation_volume_mesh_filepath(stack, downscale, label):
        fn = os.path.join(MESH_ROOTDIR, stack, "%(stack)s_down%(ds)s_annotationVolume_%(name)s_smoothed.stl" % {'stack': stack, 'name': label, 'ds':downscale})
        return fn

    @staticmethod
    def load_annotation_volume_mesh(stack, downscale, label, return_polydata_only=True):
        fn = DataManager.get_annotation_volume_mesh_filepath(stack, downscale, label)
        return load_mesh_stl(fn, return_polydata_only=return_polydata_only)

    @staticmethod
    def load_transformed_volume_meshes(stack_m, type_m, stack_f, type_f, downscale,
                                        train_sample_scheme_m=None, train_sample_scheme_f=None,
                                        global_transform_scheme=None,
                                        local_transform_scheme=None,
                                        labels=None,
                                        transitive=None,
                                        return_polydata_only=True):

        meshes = {label: DataManager.load_transformed_volume_mesh(\
        stack_m=stack_m, type_m=type_m, stack_f=stack_f, type_f=type_f, downscale=downscale,\
        train_sample_scheme_f=train_sample_scheme_f, train_sample_scheme_m=train_sample_scheme_m,\
        global_transform_scheme=global_transform_scheme, \
        local_transform_scheme=local_transform_scheme, \
        label=label, transitive=transitive, \
        return_polydata_only=return_polydata_only)
                for label in labels}

        return meshes

    @staticmethod
    def load_transformed_volume_mesh(stack_m, type_m, stack_f, type_f, downscale,
                                    train_sample_scheme_m=None, train_sample_scheme_f=None,
                                    global_transform_scheme=None,
                                    local_transform_scheme=None,
                                    label=None,
                                    transitive=None,
                                    return_polydata_only=True):
        fn = DataManager.get_transformed_volume_mesh_filepath(stack_m, type_m, stack_f, type_f, downscale,
                                            train_sample_scheme_m, train_sample_scheme_f,
                                            global_transform_scheme,
                                            local_transform_scheme,
                                            label,
                                            transitive)
        return load_mesh_stl(fn, return_polydata_only=return_polydata_only)

    @staticmethod
    def get_transformed_volume_mesh_filepath(stack_m, type_m, stack_f, type_f, downscale,
                                        train_sample_scheme_m=None, train_sample_scheme_f=None,
                                        global_transform_scheme=None,
                                        local_transform_scheme=None,
                                        label=None,
                                        transitive=None):
        basename = DataManager.get_transformed_volume_file_basename(stack_m, type_m, stack_f, type_f, downscale,
                                            train_sample_scheme_m, train_sample_scheme_f,
                                            global_transform_scheme,
                                            local_transform_scheme,
                                            label,
                                            transitive)
        return os.path.join(MESH_ROOTDIR, basename + '.stl')

    @staticmethod
    def get_transformed_volume_filepath(stack_m, type_m, stack_f, type_f, downscale,
                                        train_sample_scheme_m=None, train_sample_scheme_f=None,
                                        global_transform_scheme=None,
                                        local_transform_scheme=None,
                                        label=None, transitive=None):
        basename = DataManager.get_transformed_volume_file_basename(stack_m, type_m, stack_f, type_f, downscale,
                                            train_sample_scheme_m, train_sample_scheme_f,
                                            global_transform_scheme,
                                            local_transform_scheme,
                                            label,
                                            transitive)
        return os.path.join(VOLUME_ROOTDIR, basename + '.bp')

    @staticmethod
    def save_transformed_volume(vol, *args, **kwargs):
        volume_m_alignedTo_f_fn = DataManager.get_transformed_volume_filepath(*args, **kwargs)
        create_if_not_exists(os.path.dirname(volume_m_alignedTo_f_fn))
        bp.pack_ndarray_file(vol, volume_m_alignedTo_f_fn)

    @staticmethod
    def get_score_volume_filepath(stack, label, downscale, train_sample_scheme=None):
        suffix = generate_suffix(train_sample_scheme=train_sample_scheme)

        vol_fn = VOLUME_ROOTDIR + '/%(stack)s/score_volumes/%(stack)s_down%(ds)d_scoreVolume_%(name)s%(suffix)s.bp' % \
                {'stack':stack, 'name':label, 'ds':downscale, 'suffix':'_' + suffix if suffix != '' else ''}
        return vol_fn

    @staticmethod
    def get_score_volume_gradient_filepath_template(stack, label, downscale, train_sample_scheme=None):
        if train_sample_scheme is None:
            grad_fn = VOLUME_ROOTDIR + '/%(stack)s/score_volume_gradients/%(stack)s_down%(ds)d_scoreVolume_%(label)s_%%(suffix)s.bp' % \
                {'stack': stack, 'label': label, 'scheme':train_sample_scheme, 'ds':downscale}
        else:
            grad_fn = VOLUME_ROOTDIR + '/%(stack)s/score_volume_gradients/%(stack)s_down%(ds)d_scoreVolume_%(label)s_trainSampleScheme_%(scheme)d_%%(suffix)s.bp' % \
                {'stack': stack, 'label': label, 'scheme':train_sample_scheme, 'ds':downscale}
        return grad_fn


    @staticmethod
    def get_score_volume_gradient_filepath(stack, label, downscale, suffix, train_sample_scheme=None):

        grad_fn = VOLUME_ROOTDIR + '/%(stack)s/score_volume_gradients/%(stack)s_down%(ds)d_scoreVolume_%(label)s_trainSampleScheme_%(scheme)d_%(suffix)s.bp' % \
                {'stack': stack, 'label': label, 'scheme':train_sample_scheme, 'suffix': suffix, 'ds':downscale}

        return grad_fn

    @staticmethod
    def load_score_volume(stack, label, downscale, train_sample_scheme=None):

        vol_fn = DataManager.get_score_volume_filepath(stack=stack, label=label, downscale=downscale, train_sample_scheme=train_sample_scheme)
        score_volume = DataManager.load_data(vol_fn, filetype='bp')
        return score_volume

    @staticmethod
    def load_volume_bbox(stack, type, downscale, label=None):
        """
        annotation: with respect to aligned uncropped thumbnail
        score/thumbnail: with respect to aligned cropped thumbnail
        shell: with respect to aligned uncropped thumbnail
        """

        if type == 'annotation':
            bbox_fn = DataManager.get_annotation_volume_bbox_filepath(stack, downscale)
        elif type == 'score':
            bbox_fn = DataManager.get_score_volume_bbox_filepath(stack, label, downscale)
        elif type == 'shell':
            bbox_fn = DataManager.get_shell_bbox_filepath(stack, label, downscale)
        elif type == 'thumbnail':
            bbox_fn = DataManager.get_score_volume_bbox_filepath(stack, '7N', downscale)
        else:
            raise Exception('Type must be annotation or score.')

        volume_bbox = DataManager.load_data(bbox_fn, filetype='bbox')
        return volume_bbox

    @staticmethod
    def get_shell_bbox_filepath(stack, label, downscale):
        bbox_filepath = VOLUME_ROOTDIR + '/%(stack)s/%(stack)s_down%(ds)d_outerContourVolume_bbox.txt' % \
                        dict(stack=stack, ds=downscale)
        return bbox_filepath

    # @staticmethod
    # def load_score_volume_bbox(stack, label, downscale):
    # "use `load_volume_bbox` instead"
    #     fn = DataManager.get_score_volume_bbox_filepath(stack, label, downscale)
    #     score_volume_bbox = DataManager.load_data(bbox_fn, filetype='bbox')
    #     return score_volume_bbox

    @staticmethod
    def get_score_volume_bbox_filepath(stack, label, downscale):
        score_volume_bbox_filepath = VOLUME_ROOTDIR + '/%(stack)s/score_volumes/%(stack)s_down%(ds)d_scoreVolume_%(label)s_bbox.txt' % \
                        dict(stack=stack, ds=downscale, label=label)
        return score_volume_bbox_filepath


    @staticmethod
    def get_scoremap_viz_filepath(stack, section=None, fn=None, anchor_fn=None, structure=None, setting=None):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn): raise Exception('Section is invalid: %s.' % fn)

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        scoremap_viz_filepath = os.path.join(SCOREMAP_VIZ_ROOTDIR, '%(structure)s/%(stack)s/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_denseScoreMap_viz_setting_%(setting)d.jpg') \
            % {'stack': stack, 'fn': fn, 'structure': structure, 'anchor_fn': anchor_fn, 'setting': setting}

        return scoremap_viz_filepath


    @staticmethod
    def get_scoremap_filepath(stack, structure, setting, section=None, fn=None, anchor_fn=None, return_bbox_fp=False):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn):
                raise Exception('Section is invalid: %s.' % fn)

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        scoremap_bp_filepath = os.path.join(SCOREMAPS_ROOTDIR, \
        '%(stack)s/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_denseScoreMap_setting_%(setting)d.hdf') \
        % dict(stack=stack, fn=fn, structure=structure, anchor_fn=anchor_fn, setting=setting)

        scoremap_bbox_filepath = os.path.join(SCOREMAPS_ROOTDIR, \
        '%(stack)s/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_denseScoreMap_interpBox.txt') \
            % dict(stack=stack, fn=fn, structure=structure, anchor_fn=anchor_fn)

        if return_bbox_fp:
            return scoremap_bp_filepath, scoremap_bbox_filepath
        else:
            return scoremap_bp_filepath

    @staticmethod
    def load_scoremap(stack, section=None, fn=None, anchor_fn=None, label=None, downscale_factor=32, train_sample_scheme=None):
        """
        Return scoremaps.
        """

        # Load scoremap

        scoremap_bp_filepath, scoremap_bbox_filepath = DataManager.get_scoremap_filepath(stack, section=section, \
                                                    fn=fn, anchor_fn=anchor_fn, label=label, return_bbox_fp=True, train_sample_scheme=train_sample_scheme)
        if not os.path.exists(scoremap_bp_filepath):
            raise Exception('No scoremap for section %d for label %s\n' % (section, label))
            # return None
        scoremap = DataManager.load_data(scoremap_bp_filepath, filetype='hdf')

        # Load interpolation box
        scoremap_bbox = DataManager.load_data(scoremap_bbox_filepath, filetype='bbox')

        xmin, xmax, ymin, ymax = scoremap_bbox

        ymin_downscaled = ymin / downscale_factor
        xmin_downscaled = xmin / downscale_factor

        # full_width, full_height = DataManager.get_image_dimension(stack)
        # full_width, full_height = (16000, 13120)
        full_width, full_height = metadata_cache['image_shape'][stack]
        scoremap_downscaled = np.zeros((full_height/downscale_factor, full_width/downscale_factor), np.float32)

        # To conserve memory, it is important to make a copy of the sub-scoremap and delete the original scoremap
        scoremap_roi_downscaled = scoremap[::downscale_factor, ::downscale_factor].copy()
        del scoremap

        h_downscaled, w_downscaled = scoremap_roi_downscaled.shape

        scoremap_downscaled[ymin_downscaled : ymin_downscaled + h_downscaled,
                            xmin_downscaled : xmin_downscaled + w_downscaled] = scoremap_roi_downscaled

        return scoremap_downscaled

    @staticmethod
    def load_dnn_feature_locations(stack, section=None, fn=None, anchor_fn=None):
        fp = DataManager.get_dnn_feature_locations_filepath(stack, section=section, fn=fn, anchor_fn=anchor_fn)
        locs = np.loadtxt(fp).astype(np.int)
        indices = locs[:, 0]
        locations = locs[:, 1:]
        return indices, locations

    @staticmethod
    def get_dnn_feature_locations_filepath(stack, section=None, fn=None, anchor_fn=None):

        if section is not None:
            section_to_filename = metadata_cache['sections_to_filenames'][stack]
            fn = section_to_filename[section]

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        output_dir = create_if_not_exists(os.path.join(PATCH_FEATURES_ROOTDIR, stack,
                                       '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped' % dict(fn=fn, anchor_fn=anchor_fn)))
        output_indices_fn = os.path.join(output_dir,
                                         '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_patch_locations.txt' % \
                                         dict(fn=fn, anchor_fn=anchor_fn))
        return output_indices_fn

    @staticmethod
    def get_dnn_features_filepath(stack, section=None, fn=None, anchor_fn=None):

        if section is not None:
            section_to_filename = metadata_cache['sections_to_filenames'][stack]
            fn = section_to_filename[section]

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        feature_fn = PATCH_FEATURES_ROOTDIR + '/%(stack)s/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_features.hdf' % dict(stack=stack, fn=fn, anchor_fn=anchor_fn)
        return feature_fn

    @staticmethod
    def load_dnn_features(stack, section=None, fn=None, anchor_fn=None):
        return load_hdf(DataManager.get_dnn_features_filepath(stack, section=section, fn=fn, anchor_fn=anchor_fn))

    @staticmethod
    def get_image_filepath(stack, section=None, version='compressed', resol='lossless', data_dir=data_dir, fn=None, anchor_fn=None):
        """
        resol: can be either lossless or thumbnail
        version:
        - compressed: for regular nissl, RGB JPEG; for neurotrace, blue channel as grey JPEG
        - saturation: for regular nissl, saturation as gray, tif; for NT, blue channel as grey, tif
        - cropped: for regular nissl, lossless RGB tif; for NT, 16 bit, all channels (?) tif.
        """

        is_fluorescent = False

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn):
                raise Exception('Section is invalid: %s.' % fn)
            if (stack in all_ntb_stacks or stack in all_alt_nissl_ntb_stacks) and fn.split('-')[1][0] == 'F':
                is_fluorescent = True
        else:
            assert fn is not None

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack)

        if resol == 'thumbnail' and version == 'original_png':
            image_path = os.path.join(RAW_DATA_DIR, stack, fn + '.png')
        elif resol == 'lossless' and version == 'compressed':
            if is_fluorescent:
                image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_unsorted_alignedTo_%(anchor_fn)s_cropped_blueAsGrayscale_compressed' % {'anchor_fn':anchor_fn})
                image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_blueAsGrayscale_compressed' % {'anchor_fn':anchor_fn}])
                image_path = os.path.join(image_dir, image_name + '.jpg')
            else:
                image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_unsorted_alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn})
                image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn}])
                image_path = os.path.join(image_dir, image_name + '.jpg')
        elif resol == 'lossless' and version == 'saturation':
            if is_fluorescent: # fluorescent.
                image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_blueAsGrayscale' % {'anchor_fn':anchor_fn})
                image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_blueAsGrayscale' % {'anchor_fn':anchor_fn}])
                image_path = os.path.join(image_dir, image_name + '.tif')
            else: # Nissl
                image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_saturation' % {'anchor_fn':anchor_fn})
                image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_saturation' % {'anchor_fn':anchor_fn}])
                image_path = os.path.join(image_dir, image_name + '.tif')
        elif resol == 'lossless' and version == 'cropped':
            image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_unsorted_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn}])
            image_path = os.path.join(image_dir, image_name + '.tif')
        elif resol == 'thumbnail' and version == 'cropped_tif':
            # if stack in ['MD635']:
            #     image_dir = os.path.join(DATA_DIR, stack, stack+'_'+resol+'_unsorted_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
            #     image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn}])
            #     image_path = os.path.join(image_dir, image_name + '.tif')
            # else:
            image_dir = os.path.join(DATA_DIR, stack, stack+'_'+resol+'_unsorted_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn}])
            image_path = os.path.join(image_dir, image_name + '.tif')
        elif resol == 'thumbnail' and version == 'aligned_tif':
            image_dir = os.path.join(DATA_DIR, stack, stack+'_'+resol+'_unsorted_alignedTo_%(anchor_fn)s' % {'anchor_fn':anchor_fn})
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s' % {'anchor_fn':anchor_fn}])
            image_path = os.path.join(image_dir, image_name + '.tif')
        else:
            sys.stderr.write('Version %s and resolution %s not recognized.\n' % (version, resol))

        return image_path


    # @staticmethod
    # def get_annotation_path_v2(stack=None, username=None, timestamp='latest', orientation=None, downsample=None, annotation_rootdir=None):
    #     """Return the path to annotation."""
    #
    #     d = os.path.join(annotation_rootdir, stack)
    #
    #     if not os.path.exists(d):
    #         # sys.stderr.write('Directory %s does not exist.\n' % d)
    #         raise Exception('Directory %s does not exist.\n' % d)
    #
    #     fns = [(f, f[:-4].split('_')) for f in os.listdir(d) if f.endswith('pkl')]
    #     # stack_orient_downsample_user_timestamp.pkl
    #
    #     if username is not None:
    #         filtered_fns = [(f, f_split) for f, f_split in fns if f_split[3] == username and ((f_split[1] == orientation) if orientation is not None else True) \
    #          and ((f_split[2] == 'downsample'+str(downsample)) if downsample is not None else True)]
    #     else:
    #         filtered_fns = fns
    #
    #     if timestamp == 'latest':
    #         if len(filtered_fns) == 0:
    #             # sys.stderr.write('No annotation matches criteria.\n')
    #             # return None
    #             raise Exception('No annotation matches criteria.\n')
    #
    #         fns_sorted_by_timestamp = sorted(filtered_fns, key=lambda (f, f_split): datetime.datetime.strptime(f_split[4], "%m%d%Y%H%M%S"), reverse=True)
    #         selected_f, selected_f_split = fns_sorted_by_timestamp[0]
    #         selected_username = selected_f_split[3]
    #         selected_timestamp = selected_f_split[4]
    #     else:
    #         raise Exception('Timestamp must be `latest`.')
    #
    #     return os.path.join(d, selected_f), selected_username, selected_timestamp

    # @staticmethod
    # def load_annotation_v2(stack=None, username=None, timestamp='latest', orientation=None, downsample=None, annotation_rootdir=None):
    #     res = DataManager.get_annotation_path_v2(stack=stack, username=username, timestamp=timestamp, orientation=orientation, downsample=downsample, annotation_rootdir=annotation_rootdir)
    #     fp, usr, ts = res
    #     sys.stderr.write('Loaded annotation %s.\n' % fp)
    #     obj = pickle.load(open(fp, 'r'))
    #     if obj is None:
    #         return None
    #     else:
    #         return obj, usr, ts

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
    def get_annotation_path(stack, section, username=None, timestamp='latest', annotation_rootdir=annotation_rootdir):
        """Return the path to annotation."""

        d = os.path.join(annotation_rootdir, '%(stack)s/%(sec)04d/' % {'stack':stack, 'sec':section})

        if not os.path.exists(d):
            sys.stderr.write('Directory %s does not exist.\n' % d)
            return None

        fns = [(f, f.split('_')) for f in os.listdir(d) if f.endswith('pkl')]
        # stack_sec_user_timestamp_suffix.pkl

        if username is not None:
            filtered_fns = [(f, f_split) for f, f_split in fns if f_split[2] == username]
        else:
            filtered_fns = fns

        if timestamp == 'latest':
            if len(filtered_fns) == 0: return None
            fns_sorted_by_timestamp = sorted(filtered_fns, key=lambda (f, f_split): datetime.datetime.strptime(f_split[3], "%m%d%Y%H%M%S"), reverse=True)
            selected_f, selected_f_split = fns_sorted_by_timestamp[0]
            selected_username = selected_f_split[2]
            selected_timestamp = selected_f_split[3]
        else:
            raise Exception('Timestamp must be `latest`.')

        return os.path.join(d, selected_f), selected_username, selected_timestamp

    # @staticmethod
    # def load_annotation(stack=None, section=None, username=None, timestamp='latest', annotation_rootdir=None):
    #     res = DataManager.get_annotation_path(stack, section, username, timestamp, annotation_rootdir)
    #     if res is None:
    #         return None
    #     fp, usr, ts = res
    #     obj = pickle.load(open(fp, 'r'))
    #     if obj is None:
    #         return None
    #     else:
    #         return obj, usr, ts

    # @staticmethod
    # def save_annotation(obj, stack, section, username=None, timestamp='now', suffix='consolidated', annotation_rootdir=annotation_rootdir):
    #     d = create_if_not_exists(os.path.join(annotation_rootdir, '%(stack)s/%(sec)04d/' % {'stack':stack, 'sec':section}))
    #     fn = '_'.join([stack, '%04d'%section, username, timestamp, suffix]) + '.pkl'
    #     fp = os.path.join(d, fn)
    #     obj = pickle.dump(obj, open(fp, 'w'))
    #     return fp

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
    def convert_section_to_z(stack, sec, downsample, z_begin=None):
        """
        z_begin default to the first brainstem section.
        """

        xy_pixel_distance = xy_pixel_distance_lossless * downsample
        voxel_z_size = section_thickness / xy_pixel_distance
        # print 'voxel size:', xy_pixel_distance, xy_pixel_distance, voxel_z_size, 'um'

        # first_sec, last_sec = section_range_lookup[stack]
        first_sec, last_sec = DataManager.load_cropbox(stack)[4:]
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
        xy_pixel_distance = xy_pixel_distance_lossless * downsample
        voxel_z_size = section_thickness / xy_pixel_distance
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
        # sec_ceil = int(np.ceil(sec_float))
        # if sec_ceil == sec_floor:
        #     return sec_ceil
        # else:
        #     return sec_floor, sec_ceil

#     def __init__(self, data_dir=os.environ['DATA_DIR'],
#                  repo_dir=os.environ['REPO_DIR'],
#                  labeling_dir=os.environ['LABELING_DIR'],
#                 #  gabor_params_id=None,
#                 #  segm_params_id='tSLIC200',
#                 #  vq_params_id=None,
#                  stack=None,
#                  resol='lossless',
#                  section=None,
#                  load_mask=False):
#
#         self.data_dir = data_dir
#         self.repo_dir = repo_dir
#         self.params_dir = os.path.join(repo_dir, 'params')
#
#         self.annotation_rootdir = labeling_dir
#
#         # self.labelnames_path = os.path.join(labeling_dir, 'labelnames.txt')
#
#         # if os.path.isfile(self.labelnames_path):
#         #     with open(self.labelnames_path, 'r') as f:
#         #         self.labelnames = [n.strip() for n in f.readlines()]
#         #         self.labelnames = [n for n in self.labelnames if len(n) > 0]
#         # else:
#         #     self.labelnames = []
#
#         # self.root_results_dir = result_dir
#
#         self.slice_ind = None
#         self.image_name = None
#
#         # if gabor_params_id is None:
#         #     self.set_gabor_params('blueNisslWide')
#         # else:
#         #     self.set_gabor_params(gabor_params_id)
#         #
#         # if segm_params_id is None:
#         #     self.set_segmentation_params('blueNisslRegular')
#         # else:
#         #     self.set_segmentation_params(segm_params_id)
#         #
#         # if vq_params_id is None:
#         #     self.set_vq_params('blueNissl')
#         # else:
#         #     self.set_vq_params(vq_params_id)
#
#         if stack is not None:
#             self.set_stack(stack)
#
#         if resol is not None:
#             self.set_resol(resol)
#
#         if self.resol == 'lossless':
#             if hasattr(self, 'stack') and self.stack is not None:
#                 self.image_dir = os.path.join(self.data_dir, self.stack+'_'+self.resol+'_aligned_cropped')
#                 self.image_rgb_jpg_dir = os.path.join(self.data_dir, self.stack+'_'+self.resol+'_aligned_cropped_downscaled')
#
#         if section is not None:
#             self.set_slice(section)
#         else:
#             try:
#                 random_image_fn = os.listdir(self.image_dir)[0]
#                 self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % os.path.join(self.image_dir, random_image_fn), shell=True).split('x'))
#             except:
#                 d = os.path.join(self.data_dir, 'MD589_lossless_aligned_cropped_downscaled')
#                 if os.path.exists(d):
#                     random_image_fn = os.listdir(d)[0]
#                     self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % os.path.join(d, random_image_fn), shell=True).split('x'))
#
#         if load_mask:
#             self.thumbmail_mask = DataManager.load_data(self.data_dir+'/%(stack)s_thumbnail_aligned_cropped_mask/%(stack)s_%(slice_str)s_thumbnail_aligned_cropped_mask.png' % {'stack': self.stack, 'slice_str': self.slice_str}, filetype='image')
#             self.mask = rescale(self.thumbmail_mask.astype(np.bool), 32).astype(np.bool)
#             # self.mask[:500, :] = False
#             # self.mask[:, :500] = False
#             # self.mask[-500:, :] = False
#             # self.mask[:, -500:] = False
#
#             xs_valid = np.any(self.mask, axis=0)
#             ys_valid = np.any(self.mask, axis=1)
#             self.xmin = np.where(xs_valid)[0][0]
#             self.xmax = np.where(xs_valid)[0][-1]
#             self.ymin = np.where(ys_valid)[0][0]
#             self.ymax = np.where(ys_valid)[0][-1]
#
#             self.h = self.ymax-self.ymin+1
#             self.w = self.xmax-self.xmin+1
#
#
#     def _load_thumbnail_mask(self):
#         thumbmail_mask = DataManager.load_data(self.data_dir+'/%(stack)s_thumbnail_aligned_mask_cropped/%(stack)s_%(slice_str)s_thumbnail_aligned_mask_cropped.png' % {'stack': self.stack,
#             'slice_str': self.slice_str}, filetype='image').astype(np.bool)
#         return thumbmail_mask
#
#     def add_labelnames(self, labelnames, filename):
#         existing_labelnames = {}
#         with open(filename, 'r') as f:
#             for ln in f.readlines():
#                 abbr, fullname = ln.split('\t')
#                 existing_labelnames[abbr] = fullname.strip()
#
#         with open(filename, 'a') as f:
#             for abbr, fullname in labelnames.iteritems():
#                 if abbr not in existing_labelnames:
#                     f.write(abbr+'\t'+fullname+'\n')
#
#     def set_stack(self, stack):
#         self.stack = stack
#         self._get_image_dimension()
# #         self.stack_path = os.path.join(self.data_dir, self.stack)
# #         self.slice_ind = None
#
#     def set_resol(self, resol):
#         self.resol = resol
#
#     def _get_image_dimension(self):
#
#         try:
#             if hasattr(self, 'image_path') and os.path.exists(self.image_path):
#                 self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self.image_path, shell=True).split('x'))
#             else:
#                 # sys.stderr.write('original TIFF image is not available. Loading downscaled jpg instead...')
#
#                 # if section is specified, use that section; otherwise use a random section in the brainstem range
#                 if hasattr(self, 'slice_ind') and self.slice_ind is not None:
#                     sec = self.slice_ind
#                 else:
#                     sec = section_range_lookup[self.stack][0]
#
#                 self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self._get_image_filepath(section=sec, version='rgb-jpg'), shell=True).split('x'))
#
#         except Exception as e:
#             print e
#             sys.stderr.write('Cannot find image. Make sure the data folder is mounted.\n')
#
#         return self.image_width, self.image_height
#
#     def set_slice(self, slice_ind):
#         assert self.stack is not None and self.resol is not None, 'Stack is not specified'
#         self.slice_ind = slice_ind
#         self.slice_str = '%04d' % slice_ind
#         if self.resol == 'lossless':
#             self.image_dir = os.path.join(self.data_dir, self.stack+'_'+self.resol+'_aligned_cropped')
#             self.image_name = '_'.join([self.stack, self.slice_str, self.resol])
#             self.image_path = os.path.join(self.image_dir, self.image_name + '_aligned_cropped.tif')
#
#         try:
#             if os.path.exists(self.image_path):
#                 self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self.image_path, shell=True).split('x'))
#             else:
#                 # sys.stderr.write('original TIFF image is not available. Loading downscaled jpg instead...')
#                 self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self._get_image_filepath(version='rgb-jpg'), shell=True).split('x'))
#         except Exception as e:
#             print e
#             sys.stderr.write('Cannot find image\n')
#
#         # self.labelings_dir = os.path.join(self.image_dir, 'labelings')
#
#         # if hasattr(self, 'result_list'):
#         #     del self.result_list
#
#         # self.labelings_dir = os.path.join(self.root_labelings_dir, self.stack, self.slice_str)
#         # if not os.path.exists(self.labelings_dir):
#         #     os.makedirs(self.labelings_dir)
#
# #         self.results_dir = os.path.join(self.image_dir, 'pipelineResults')
#
#         # self.results_dir = os.path.join(self.root_results_dir, self.stack, self.slice_str)
#         # if not os.path.exists(self.results_dir):
#         #     os.makedirs(self.results_dir)
#
#     def _get_annotation_path(self, stack=None, section=None, username=None, timestamp='latest'):
#         if stack is None:
#             stack = self.stack
#         if section is None:
#             section = self.slice_ind
#         return DataManager.get_annotation_path(stack, section, username, timestamp,
#                                             annotation_rootdir=self.annotation_rootdir)
#
#     def _load_annotation(self, stack=None, section=None, username=None, timestamp='latest'):
#         if stack is None:
#             stack = self.stack
#         if section is None:
#             section = self.slice_ind
#         return DataManager.load_annotation(stack, section, username, timestamp, self.annotation_rootdir)
#
#     def _save_annotation(self, obj, username=None, timestamp='now', suffix='consolidated', annotation_rootdir=None):
#         if annotation_rootdir is None:
#             annotation_rootdir = self.annotation_rootdir
#         if stack is None:
#             stack = self.stack
#         if section is None:
#             section = self.slice_ind
#         return DataManager.save_annotation(stack, section, username, timestamp, suffix, annotation_rootdir)
#
#     def _get_image_filepath(self, stack=None, resol='lossless', section=None, version='rgb-jpg'):
#         if stack is None:
#             stack = self.stack
#         if resol is None:
#             resol = self.resol
#         if section is None:
#             section = self.slice_ind
#
#         slice_str = '%04d' % section
#
#         if version == 'rgb-jpg':
#             image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_aligned_cropped_downscaled')
#             image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_downscaled'])
#             image_path = os.path.join(image_dir, image_name + '.jpg')
#         # elif version == 'gray-jpg':
#         #     image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_cropped_grayscale_downscaled')
#         #     image_name = '_'.join([stack, slice_str, resol, 'warped'])
#         #     image_path = os.path.join(image_dir, image_name + '.jpg')
#         elif version == 'gray':
#             image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_aligned_cropped_grayscale')
#             image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_grayscale'])
#             image_path = os.path.join(image_dir, image_name + '.tif')
#         elif version == 'rgb':
#             image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_aligned_cropped')
#             image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped'])
#             image_path = os.path.join(image_dir, image_name + '.tif')
#
#         elif version == 'stereotactic-rgb-jpg':
#             image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_aligned_cropped_downscaled_stereotactic')
#             image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_downscaled_stereotactic'])
#             image_path = os.path.join(image_dir, image_name + '.jpg')
#
#         return image_path
#
#     def _read_image(self, image_filename):
#         if image_filename.endswith('tif') or image_filename.endswith('tiff'):
#             from PIL.Image import open
#             img = np.array(open(image_filename))/255.
#         else:
#             img = DataManager.load_data(image_filename, filetype='image')
#         return img
#
#     def _load_image(self, versions=['rgb', 'gray', 'rgb-jpg'], force_reload=True):
#
#         assert self.image_name is not None, 'Image is not specified'
#
#         if 'rgb-jpg' in versions:
#             if force_reload or not hasattr(self, 'image_rgb_jpg'):
#                 image_filename = self._get_image_filepath(version='rgb-jpg')
#                 # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
#                 self.image_rgb_jpg = self._read_image(image_filename)
#
#         if 'rgb' in versions:
#             if force_reload or not hasattr(self, 'image_rgb'):
#                 image_filename = self._get_image_filepath(version='rgb')
#                 # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
#                 self.image_rgb = self._read_image(image_filename)
#
#         if 'gray' in versions and not hasattr(self, 'image'):
#             if force_reload or not hasattr(self, 'gray'):
#                 image_filename = self._get_image_filepath(version='gray')
#                 # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
#                 self.image = self._read_image(image_filename)
#
#     def _regulate_image(self, img, is_rgb=None):
#         """
#         Ensure the image is of type uint8.
#         """
#
#         if not np.issubsctype(img, np.uint8):
#             try:
#                 img = img_as_ubyte(img)
#             except:
#                 img_norm = (img-img.min()).astype(np.float)/(img.max() - img.min())
#                 img = img_as_ubyte(img_norm)
#
#         if is_rgb is not None:
#             if img.ndim == 2 and is_rgb:
#                 img = gray2rgb(img)
#             elif img.ndim == 3 and not is_rgb:
#                 img = rgb2gray(img)
#
#         return img

##################################################

# This module stores any meta information that is dynamic.
metadata_cache = {}
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
 'MD642': (28704, 15584)}
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
