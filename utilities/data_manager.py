from utilities2015 import *
from metadata import *

class DataManager(object):

    @staticmethod
    def load_volume_label_to_name(stack):
        label_to_name = {}
        name_to_label = {}

        with open(os.path.join(volume_dir, stack, stack+'_down32_annotationVolume_nameToLabel.txt'), 'r') as f:
            for line in f.readlines():
                name_s, label = line.split()
                label_to_name[int(label)] = name_s
                name_to_label[name_s] = int(label)

        return label_to_name, name_to_label

    @staticmethod
    def load_volume_bbox(stack):
        with open(os.path.join(volume_dir, stack, stack+'_down32_annotationVolume_bbox.txt'), 'r') as f:
            bbox = map(int, f.readline().strip().split())
        return bbox

    @staticmethod
    def load_anchor_filename(stack):
        with open(thumbnail_data_dir + '/%(stack)s/%(stack)s_anchor.txt'%dict(stack=stack), 'r') as f:
            anchor_fn = f.readline()
        return anchor_fn

    @staticmethod
    def load_cropbox(stack):
        with open(thumbnail_data_dir + '/%(stack)s/%(stack)s_cropbox.txt'%dict(stack=stack), 'r') as f:
            cropbox = one_liner_to_arr(f.readline(), int)
        return cropbox

    @staticmethod
    def load_sorted_filenames(stack):
        with open(thumbnail_data_dir + '/%(stack)s/%(stack)s_sorted_filenames.txt'%dict(stack=stack), 'r') as f:
            fn_idx_tuples = [line.strip().split() for line in f.readlines()]
            filename_to_section = {fn: int(idx) for fn, idx in fn_idx_tuples}
            section_to_filename = {int(idx): fn for fn, idx in fn_idx_tuples}
        return filename_to_section, section_to_filename

    @staticmethod
    def load_transforms(stack, downsample_factor):

        import cPickle as pickle
        Ts = pickle.load(open(thumbnail_data_dir + '/%(stack)s/%(stack)s_elastix_output/%(stack)s_transformsTo_anchor.pkl' % dict(stack=stack), 'r'))

        Ts_inv_downsampled = {}
        for fn, T0 in Ts.iteritems():
            T = T0.copy()
            T[:2, 2] = T[:2, 2] * 32 / downsample_factor
            Tinv = np.linalg.inv(T)
            Ts_inv_downsampled[fn] = Tinv

        return Ts_inv_downsampled

    @staticmethod
    def save_thumbnail_mask(mask, stack, section, cerebellum_removed=False):

        fn = DataManager.get_thumbnail_mask_filepath(stack, section, cerebellum_removed=cerebellum_removed)
        create_if_not_exists(os.path.dirname(fn))
        imsave(fn, mask)
        sys.stderr.write('Thumbnail mask for section %s, %d saved to %s.\n' % (stack, section, fn))

    @staticmethod
    def load_thumbnail_mask(stack, section, cerebellum_removed=False):
        fn = DataManager.get_thumbnail_mask_filepath(stack, section, cerebellum_removed=cerebellum_removed)
        thumbmail_mask = imread(fn).astype(np.bool)
        return thumbmail_mask

    @staticmethod
    def load_thumbnail_mask_v2(stack, section=None, version='aligned_cropped'):
        anchor_fn = DataManager.load_anchor_filename(stack)
        filename_to_section, section_to_filename = DataManager.load_sorted_filenames(stack)
        fn = section_to_filename[section]

        if version == 'aligned_cropped':
            #
            # image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_unsorted_alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn})
            # image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn}])
            # image_path = os.path.join(image_dir, image_name + '.jpg')

            fn = data_dir+'/%(stack)s/%(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s_cropped/%(fn)s_mask_alignedTo_%(anchor_fn)s_cropped.png' % \
                dict(stack=stack, fn=fn, anchor_fn=anchor_fn)
        elif version == 'aligned':
            fn = data_dir+'/%(stack)s/%(stack)s_mask_unsorted_alignedTo_%(anchor_fn)s/%(stack)s_%(sec)04d_mask_alignedTo_%(anchor_fn)s.png' % \
                dict(stack=stack, fn=fn, anchor_fn=anchor_fn)

        # if version == 'aligned_cropped':
        #     fn = data_dir+'/%(stack)s/%(stack)s_mask_unsorted_aligned_cropped/%(stack)s_%(sec)04d_mask_aligned_cropped.png' % \
        #         dict(stack=stack, sec=section)
        # elif version == 'aligned':
        #     fn = data_dir+'/%(stack)s/%(stack)s_mask_sorted_aligned/%(stack)s_%(sec)04d_mask_aligned.png' % \
        #         dict(stack=stack, sec=section)

        mask = imread(fn).astype(np.bool)
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
    def get_global_alignment_parameters_filepath_prefix(stack_fixed, stack_moving, fixed_volume_type='score', moving_volume_type='score', train_sample_scheme=1, global_transform_scheme=1):

        atlasAlignParams_dir = atlasAlignParams_rootdir + '/%(stack_moving)s_to_%(stack_fixed)s' % \
                     {'stack_moving': stack_moving, 'stack_fixed': stack_fixed}

        def type_to_str(t):
            if t == 'score':
                return 'scoreVolume'
            elif t == 'annotation':
                return 'annotationVolume'

        return atlasAlignParams_dir + '/%(stack_moving)s_down32_%(m_str)s_to_%(stack_fixed)s_down32_%(f_str)s_trainSampleScheme_%(scheme)d_globalTxScheme_%(gtf_sheme)d' % \
                  {'stack_moving': stack_moving, 'stack_fixed': stack_fixed,
                  'm_str': type_to_str(moving_volume_type), 'f_str': type_to_str(fixed_volume_type),
                  'scheme':train_sample_scheme, 'gtf_sheme':global_transform_scheme}

    @staticmethod
    def get_global_alignment_parameters_filepath(stack_fixed, stack_moving, fixed_volume_type='score', moving_volume_type='score', train_sample_scheme=1, global_transform_scheme=1):
        partial_fn = DataManager.get_global_alignment_parameters_filepath_prefix(stack_fixed, stack_moving, fixed_volume_type, moving_volume_type, train_sample_scheme, global_transform_scheme)
        return partial_fn + '_parameters.txt'

    @staticmethod
    def get_global_alignment_score_plot_filepath(stack_fixed, stack_moving, fixed_volume_type='score', moving_volume_type='score', train_sample_scheme=1, global_transform_scheme=1):
        partial_fn = DataManager.get_global_alignment_parameters_filepath_prefix(stack_fixed, stack_moving, fixed_volume_type, moving_volume_type, train_sample_scheme, global_transform_scheme)
        return partial_fn + '_scoreEvolution.png'

    @staticmethod
    def get_global_alignment_viz_filepath(stack_fixed, stack_moving, fixed_volume_type='score', moving_volume_type='score', train_sample_scheme=1, global_transform_scheme=1):

        atlasAlignParams_dir = atlasAlignParams_rootdir + '/%(stack_moving)s_to_%(stack_fixed)s' % \
                     {'stack_moving': stack_moving, 'stack_fixed': stack_fixed}

        def type_to_str(t):
            if t == 'score':
                return 'scoreVolume'
            elif t == 'annotation':
                return 'annotationVolume'

        # return atlasAlignParams_dir + '/%(stack_moving)s_down32_%(m_str)s_to_%(stack_fixed)s_down32_%(f_str)s_trainSampleScheme_%(scheme)d_globalTxScheme_%(gtf_sheme)d' % \
        #           {'stack_moving': stack_moving, 'stack_fixed': stack_fixed,
        #           'm_str': type_to_str(moving_volume_type), 'f_str': type_to_str(fixed_volume_type),
        #           'scheme':train_sample_scheme, 'gtf_sheme':global_transform_scheme}

        viz_dir = atlasAlignParams_dir + '/%(stack_moving)s_down32_%(m_str)s_to_%(stack_fixed)s_down32_%(f_str)s_trainSampleScheme_%(scheme)d_globalTxScheme_%(gtf_sheme)d_viz' % \
                    {'stack_moving': stack_moving, 'stack_fixed': stack_fixed,
                    'm_str': type_to_str(moving_volume_type), 'f_str': type_to_str(fixed_volume_type),
                    'scheme':train_sample_scheme, 'gtf_sheme':global_transform_scheme}
        return viz_dir


    @staticmethod
    def get_svm_filepath(label, suffix=''):
        return SVM_ROOTDIR + '/classifiers/%(label)s_svm_%(suffix)s.pkl' % {'label': label, 'suffix':suffix}

    @staticmethod
    def get_sparse_scores_filepath(stack, fn, anchor_fn, label, suffix):
        # sparse_score_dir = os.path.join(SPARSE_SCORES_ROOTDIR, stack, '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped' % \
        #                               {'fn': fn, 'anchor_fn': anchor_fn})
        # return sparse_score_dir + '/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(label)s_sparseScores%(suffix)s.hdf' % \
        #             {'fn': fn, 'anchor_fn': anchor_fn, 'label':label, 'suffix': '_'+suffix if suffix != '' else ''}

        return os.path.join(SPARSE_SCORES_ROOTDIR, stack, '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped', \
        '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(label)s_sparseScores_%(suffix)s.hdf') % \
        {'fn': fn, 'anchor_fn': anchor_fn, 'label':label, 'suffix': suffix}

    @staticmethod
    def get_score_volume_filepath(stack, label, downscale, suffix):
        vol_fn = VOLUME_ROOTDIR + '/%(stack)s/score_volumes/%(stack)s_down%(ds)d_scoreVolume_%(name)s_%(suffix)s.bp' % \
                {'stack':stack, 'name':label, 'ds':downscale, 'suffix':suffix}
        return vol_fn

    @staticmethod
    def load_score_volume(stack, label, downscale, suffix=''):

        vol_fn = DataManager.get_score_volume_filepath(stack, label, downscale, suffix)
        score_volume = bp.unpack_ndarray_file(vol_fn)
        return score_volume


    @staticmethod
    def get_score_volume_bbox_filepath(stack, label, downscale):
        score_volume_bbox_filepath = VOLUME_ROOTDIR + '/%(stack)s/score_volumes/%(stack)s_down%(ds)d_scoreVolume_%(label)s_bbox.txt' % \
                        dict(stack=stack, ds=downscale, label=label)
        return score_volume_bbox_filepath


    @staticmethod
    def get_scoremap_viz_filepath(stack, section=None, fn=None, anchor_fn=None, label=None, train_sample_scheme=1):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if fn in ['Nonexisting', 'Rescan', 'Placeholder']:
                raise Exception('Section is invalid: %s.' % fn)

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        scoremap_viz_filepath = SCOREMAP_VIZ_ROOTDIR + '/%(label)s/%(stack)s/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(label)s_denseScoreMap_viz_trainSampleScheme_%(scheme)d.jpg' \
            % {'stack': stack, 'fn': fn, 'label': label, 'anchor_fn': anchor_fn, 'scheme': train_sample_scheme}

        return scoremap_viz_filepath


    @staticmethod
    def get_scoremap_filepath(stack, section=None, fn=None, anchor_fn=None, label=None, return_bbox_fp=False, train_sample_scheme=1):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if fn in ['Nonexisting', 'Rescan', 'Placeholder']:
                raise Exception('Section is invalid: %s.' % fn)

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        scoremap_bp_filepath = SCOREMAPS_ROOTDIR + '/%(stack)s/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(label)s_denseScoreMap_trainSampleScheme_%(scheme)d.hdf' \
        % {'stack': stack, 'fn': fn, 'label': label, 'anchor_fn': anchor_fn, 'scheme':train_sample_scheme}

        scoremap_bbox_filepath = SCOREMAPS_ROOTDIR + \
        '/%(stack)s/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(label)s_denseScoreMap_interpBox.txt' \
            % dict(stack=stack, fn=fn, label=label, anchor_fn=anchor_fn)

        if return_bbox_fp:
            return scoremap_bp_filepath, scoremap_bbox_filepath
        else:
            return scoremap_bp_filepath

    @staticmethod
    def load_scoremap(stack, section=None, fn=None, anchor_fn=None, label=None, downscale_factor=32, suffix=''):

        # Load scoremap

        scoremap_bp_filepath, scoremap_bbox_filepath = DataManager.get_scoremap_filepath(stack, section=section, \
                                                    fn=fn, anchor_fn=anchor_fn, label=label, return_bbox_fp=True, suffix=suffix)
        if not os.path.exists(scoremap_bp_filepath):
            raise Exception('No scoremap for section %d for label %s\n' % (section, label))
            # return None
        scoremap = load_hdf(scoremap_bp_filepath)

        # Load interpolation box
        scoremap_bbox = np.loadtxt(scoremap_bbox_filepath).astype(np.int)

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
    def get_image_filepath(stack, section=None, version='rgb-jpg', resol='lossless', data_dir=data_dir, fn=None, anchor_fn=None):

        if section is not None:
            _, section_to_filename = DataManager.load_sorted_filenames(stack)
            fn = section_to_filename[section]
            if fn in ['Nonexisting', 'Rescan', 'Placeholder']:
                raise Exception('Section is invalid: %s.' % fn)

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack)

        if resol == 'lossless' and version == 'rgb-jpg':
            image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_unsorted_alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn})
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn}])
            image_path = os.path.join(image_dir, image_name + '.jpg')
        elif resol == 'lossless' and version == 'saturation':
            image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_unsorted_alignedTo_%(anchor_fn)s_cropped_saturation' % {'anchor_fn':anchor_fn})
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_saturation' % {'anchor_fn':anchor_fn}])
            image_path = os.path.join(image_dir, image_name + '.jpg')
        elif resol == 'thumbnail' and version == 'cropped_tif':
            image_dir = os.path.join(DATA_DIR, stack, stack+'_'+resol+'_unsorted_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn}])
            image_path = os.path.join(image_dir, image_name + '.tif')
        elif resol == 'thumbnail' and version == 'aligned_tif':
            image_dir = os.path.join(DATA_DIR, stack, stack+'_'+resol+'_unsorted_alignedTo_%(anchor_fn)s' % {'anchor_fn':anchor_fn})
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s' % {'anchor_fn':anchor_fn}])
            image_path = os.path.join(image_dir, image_name + '.tif')

        return image_path


    @staticmethod
    def get_annotation_path_v2(stack=None, username=None, timestamp='latest', orientation=None, downsample=None, annotation_rootdir=None):
        """Return the path to annotation."""

        d = os.path.join(annotation_rootdir, stack)

        if not os.path.exists(d):
            # sys.stderr.write('Directory %s does not exist.\n' % d)
            raise Exception('Directory %s does not exist.\n' % d)

        fns = [(f, f[:-4].split('_')) for f in os.listdir(d) if f.endswith('pkl')]
        # stack_orient_downsample_user_timestamp.pkl

        if username is not None:
            filtered_fns = [(f, f_split) for f, f_split in fns if f_split[3] == username and ((f_split[1] == orientation) if orientation is not None else True) \
             and ((f_split[2] == 'downsample'+str(downsample)) if downsample is not None else True)]
        else:
            filtered_fns = fns

        if timestamp == 'latest':
            if len(filtered_fns) == 0:
                # sys.stderr.write('No annotation matches criteria.\n')
                # return None
                raise Exception('No annotation matches criteria.\n')

            fns_sorted_by_timestamp = sorted(filtered_fns, key=lambda (f, f_split): datetime.datetime.strptime(f_split[4], "%m%d%Y%H%M%S"), reverse=True)
            selected_f, selected_f_split = fns_sorted_by_timestamp[0]
            selected_username = selected_f_split[3]
            selected_timestamp = selected_f_split[4]
        else:
            raise Exception('Timestamp must be `latest`.')

        return os.path.join(d, selected_f), selected_username, selected_timestamp

    @staticmethod
    def load_annotation_v2(stack=None, username=None, timestamp='latest', orientation=None, downsample=None, annotation_rootdir=None):
        res = DataManager.get_annotation_path_v2(stack=stack, username=username, timestamp=timestamp, orientation=orientation, downsample=downsample, annotation_rootdir=annotation_rootdir)
        fp, usr, ts = res
        sys.stderr.write('Loaded annotation %s.\n' % fp)
        obj = pickle.load(open(fp, 'r'))
        if obj is None:
            return None
        else:
            return obj, usr, ts


    @staticmethod
    def load_annotation_v3(stack=None, annotation_rootdir=None):
        from pandas import read_hdf
        fn = os.path.join(annotation_rootdir, stack, '%(stack)s_annotation_v3.h5' % {'stack':stack})
        contour_df = read_hdf(fn, 'contours')
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

    @staticmethod
    def load_annotation(stack=None, section=None, username=None, timestamp='latest', annotation_rootdir=None):
        res = DataManager.get_annotation_path(stack, section, username, timestamp, annotation_rootdir)
        if res is None:
            return None
        fp, usr, ts = res
        obj = pickle.load(open(fp, 'r'))
        if obj is None:
            return None
        else:
            return obj, usr, ts

    @staticmethod
    def save_annotation(obj, stack, section, username=None, timestamp='now', suffix='consolidated', annotation_rootdir=annotation_rootdir):
        d = create_if_not_exists(os.path.join(annotation_rootdir, '%(stack)s/%(sec)04d/' % {'stack':stack, 'sec':section}))
        fn = '_'.join([stack, '%04d'%section, username, timestamp, suffix]) + '.pkl'
        fp = os.path.join(d, fn)
        obj = pickle.dump(obj, open(fp, 'w'))
        return fp

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
            fn = DataManager.get_image_filepath(stack=stack, version='rgb-jpg', data_dir=data_dir, fn=random_fn, anchor_fn=anchor_fn)
            if not os.path.exists(fn):
                fn = DataManager.get_image_filepath(stack=stack, version='saturation', data_dir=data_dir, fn=random_fn, anchor_fn=anchor_fn)
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

    def __init__(self, data_dir=os.environ['DATA_DIR'],
                 repo_dir=os.environ['REPO_DIR'],
                 labeling_dir=os.environ['LABELING_DIR'],
                #  gabor_params_id=None,
                #  segm_params_id='tSLIC200',
                #  vq_params_id=None,
                 stack=None,
                 resol='lossless',
                 section=None,
                 load_mask=False):

        self.data_dir = data_dir
        self.repo_dir = repo_dir
        self.params_dir = os.path.join(repo_dir, 'params')

        self.annotation_rootdir = labeling_dir

        # self.labelnames_path = os.path.join(labeling_dir, 'labelnames.txt')

        # if os.path.isfile(self.labelnames_path):
        #     with open(self.labelnames_path, 'r') as f:
        #         self.labelnames = [n.strip() for n in f.readlines()]
        #         self.labelnames = [n for n in self.labelnames if len(n) > 0]
        # else:
        #     self.labelnames = []

        # self.root_results_dir = result_dir

        self.slice_ind = None
        self.image_name = None

        # if gabor_params_id is None:
        #     self.set_gabor_params('blueNisslWide')
        # else:
        #     self.set_gabor_params(gabor_params_id)
        #
        # if segm_params_id is None:
        #     self.set_segmentation_params('blueNisslRegular')
        # else:
        #     self.set_segmentation_params(segm_params_id)
        #
        # if vq_params_id is None:
        #     self.set_vq_params('blueNissl')
        # else:
        #     self.set_vq_params(vq_params_id)

        if stack is not None:
            self.set_stack(stack)

        if resol is not None:
            self.set_resol(resol)

        if self.resol == 'lossless':
            if hasattr(self, 'stack') and self.stack is not None:
                self.image_dir = os.path.join(self.data_dir, self.stack+'_'+self.resol+'_aligned_cropped')
                self.image_rgb_jpg_dir = os.path.join(self.data_dir, self.stack+'_'+self.resol+'_aligned_cropped_downscaled')

        if section is not None:
            self.set_slice(section)
        else:
            try:
                random_image_fn = os.listdir(self.image_dir)[0]
                self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % os.path.join(self.image_dir, random_image_fn), shell=True).split('x'))
            except:
                d = os.path.join(self.data_dir, 'MD589_lossless_aligned_cropped_downscaled')
                if os.path.exists(d):
                    random_image_fn = os.listdir(d)[0]
                    self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % os.path.join(d, random_image_fn), shell=True).split('x'))

        if load_mask:
            self.thumbmail_mask = imread(self.data_dir+'/%(stack)s_thumbnail_aligned_cropped_mask/%(stack)s_%(slice_str)s_thumbnail_aligned_cropped_mask.png' % {'stack': self.stack, 'slice_str': self.slice_str})
            self.mask = rescale(self.thumbmail_mask.astype(np.bool), 32).astype(np.bool)
            # self.mask[:500, :] = False
            # self.mask[:, :500] = False
            # self.mask[-500:, :] = False
            # self.mask[:, -500:] = False

            xs_valid = np.any(self.mask, axis=0)
            ys_valid = np.any(self.mask, axis=1)
            self.xmin = np.where(xs_valid)[0][0]
            self.xmax = np.where(xs_valid)[0][-1]
            self.ymin = np.where(ys_valid)[0][0]
            self.ymax = np.where(ys_valid)[0][-1]

            self.h = self.ymax-self.ymin+1
            self.w = self.xmax-self.xmin+1


    def _load_thumbnail_mask(self):
        thumbmail_mask = imread(self.data_dir+'/%(stack)s_thumbnail_aligned_mask_cropped/%(stack)s_%(slice_str)s_thumbnail_aligned_mask_cropped.png' % {'stack': self.stack,
            'slice_str': self.slice_str}).astype(np.bool)
        return thumbmail_mask

    def add_labelnames(self, labelnames, filename):
        existing_labelnames = {}
        with open(filename, 'r') as f:
            for ln in f.readlines():
                abbr, fullname = ln.split('\t')
                existing_labelnames[abbr] = fullname.strip()

        with open(filename, 'a') as f:
            for abbr, fullname in labelnames.iteritems():
                if abbr not in existing_labelnames:
                    f.write(abbr+'\t'+fullname+'\n')

    def set_stack(self, stack):
        self.stack = stack
        self._get_image_dimension()
#         self.stack_path = os.path.join(self.data_dir, self.stack)
#         self.slice_ind = None

    def set_resol(self, resol):
        self.resol = resol

    def _get_image_dimension(self):

        try:
            if hasattr(self, 'image_path') and os.path.exists(self.image_path):
                self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self.image_path, shell=True).split('x'))
            else:
                # sys.stderr.write('original TIFF image is not available. Loading downscaled jpg instead...')

                # if section is specified, use that section; otherwise use a random section in the brainstem range
                if hasattr(self, 'slice_ind') and self.slice_ind is not None:
                    sec = self.slice_ind
                else:
                    sec = section_range_lookup[self.stack][0]

                self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self._get_image_filepath(section=sec, version='rgb-jpg'), shell=True).split('x'))

        except Exception as e:
            print e
            sys.stderr.write('Cannot find image. Make sure the data folder is mounted.\n')

        return self.image_width, self.image_height

    def set_slice(self, slice_ind):
        assert self.stack is not None and self.resol is not None, 'Stack is not specified'
        self.slice_ind = slice_ind
        self.slice_str = '%04d' % slice_ind
        if self.resol == 'lossless':
            self.image_dir = os.path.join(self.data_dir, self.stack+'_'+self.resol+'_aligned_cropped')
            self.image_name = '_'.join([self.stack, self.slice_str, self.resol])
            self.image_path = os.path.join(self.image_dir, self.image_name + '_aligned_cropped.tif')

        try:
            if os.path.exists(self.image_path):
                self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self.image_path, shell=True).split('x'))
            else:
                # sys.stderr.write('original TIFF image is not available. Loading downscaled jpg instead...')
                self.image_width, self.image_height = map(int, check_output("identify -format %%Wx%%H %s" % self._get_image_filepath(version='rgb-jpg'), shell=True).split('x'))
        except Exception as e:
            print e
            sys.stderr.write('Cannot find image\n')

        # self.labelings_dir = os.path.join(self.image_dir, 'labelings')

        # if hasattr(self, 'result_list'):
        #     del self.result_list

        # self.labelings_dir = os.path.join(self.root_labelings_dir, self.stack, self.slice_str)
        # if not os.path.exists(self.labelings_dir):
        #     os.makedirs(self.labelings_dir)

#         self.results_dir = os.path.join(self.image_dir, 'pipelineResults')

        # self.results_dir = os.path.join(self.root_results_dir, self.stack, self.slice_str)
        # if not os.path.exists(self.results_dir):
        #     os.makedirs(self.results_dir)

    def _get_annotation_path(self, stack=None, section=None, username=None, timestamp='latest'):
        if stack is None:
            stack = self.stack
        if section is None:
            section = self.slice_ind
        return DataManager.get_annotation_path(stack, section, username, timestamp,
                                            annotation_rootdir=self.annotation_rootdir)

    def _load_annotation(self, stack=None, section=None, username=None, timestamp='latest'):
        if stack is None:
            stack = self.stack
        if section is None:
            section = self.slice_ind
        return DataManager.load_annotation(stack, section, username, timestamp, self.annotation_rootdir)

    def _save_annotation(self, obj, username=None, timestamp='now', suffix='consolidated', annotation_rootdir=None):
        if annotation_rootdir is None:
            annotation_rootdir = self.annotation_rootdir
        if stack is None:
            stack = self.stack
        if section is None:
            section = self.slice_ind
        return DataManager.save_annotation(stack, section, username, timestamp, suffix, annotation_rootdir)

    def _get_image_filepath(self, stack=None, resol='lossless', section=None, version='rgb-jpg'):
        if stack is None:
            stack = self.stack
        if resol is None:
            resol = self.resol
        if section is None:
            section = self.slice_ind

        slice_str = '%04d' % section

        if version == 'rgb-jpg':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_aligned_cropped_downscaled')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_downscaled'])
            image_path = os.path.join(image_dir, image_name + '.jpg')
        # elif version == 'gray-jpg':
        #     image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_cropped_grayscale_downscaled')
        #     image_name = '_'.join([stack, slice_str, resol, 'warped'])
        #     image_path = os.path.join(image_dir, image_name + '.jpg')
        elif version == 'gray':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_aligned_cropped_grayscale')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_grayscale'])
            image_path = os.path.join(image_dir, image_name + '.tif')
        elif version == 'rgb':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_aligned_cropped')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped'])
            image_path = os.path.join(image_dir, image_name + '.tif')

        elif version == 'stereotactic-rgb-jpg':
            image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_aligned_cropped_downscaled_stereotactic')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_downscaled_stereotactic'])
            image_path = os.path.join(image_dir, image_name + '.jpg')

        return image_path

    def _read_image(self, image_filename):
        if image_filename.endswith('tif') or image_filename.endswith('tiff'):
            from PIL.Image import open
            img = np.array(open(image_filename))/255.
        else:
            img = imread(image_filename)
        return img

    def _load_image(self, versions=['rgb', 'gray', 'rgb-jpg'], force_reload=True):

        assert self.image_name is not None, 'Image is not specified'

        if 'rgb-jpg' in versions:
            if force_reload or not hasattr(self, 'image_rgb_jpg'):
                image_filename = self._get_image_filepath(version='rgb-jpg')
                # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
                self.image_rgb_jpg = self._read_image(image_filename)

        if 'rgb' in versions:
            if force_reload or not hasattr(self, 'image_rgb'):
                image_filename = self._get_image_filepath(version='rgb')
                # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
                self.image_rgb = self._read_image(image_filename)

        if 'gray' in versions and not hasattr(self, 'image'):
            if force_reload or not hasattr(self, 'gray'):
                image_filename = self._get_image_filepath(version='gray')
                # assert os.path.exists(image_filename), "Image '%s' does not exist" % (self.image_name + '.tif')
                self.image = self._read_image(image_filename)

    def _regulate_image(self, img, is_rgb=None):
        """
        Ensure the image is of type uint8.
        """

        if not np.issubsctype(img, np.uint8):
            try:
                img = img_as_ubyte(img)
            except:
                img_norm = (img-img.min()).astype(np.float)/(img.max() - img.min())
                img = img_as_ubyte(img_norm)

        if is_rgb is not None:
            if img.ndim == 2 and is_rgb:
                img = gray2rgb(img)
            elif img.ndim == 3 and not is_rgb:
                img = rgb2gray(img)

        return img

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
 'MD603': (20928, 13472)}
metadata_cache['anchor_fn'] = {stack: DataManager.load_anchor_filename(stack) for stack in all_stacks}
metadata_cache['sections_to_filenames'] = {stack: DataManager.load_sorted_filenames(stack)[1] for stack in all_stacks}
metadata_cache['section_limits'] = {stack: DataManager.load_cropbox(stack)[4:] for stack in all_stacks}
