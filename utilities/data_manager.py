import sys
import os
import subprocess

from pandas import read_hdf

sys.path.append(os.path.join(os.environ['REPO_DIR'], 'utilities'))
from utilities2015 import *
from metadata import *
from vis3d_utilities import *
from distributed_utilities import *

def get_random_masked_regions(region_shape, stack, num_regions=1, sec=None, fn=None):
    """
    Return a random region that is on mask.

    Args:
        region_shape: (width, height)

    Returns:
        list of (region_x, region_y, region_w, region_h)
    """

    if fn is None:
        fn = metadata_cache['sections_to_filenames'][stack][sec]
    tb_mask = DataManager.load_thumbnail_mask_v2(stack=stack, fn=fn)
    img_w, img_h = metadata_cache['image_shape'][stack]
    h, w = region_shape

    regions = []
    for _ in range(num_regions):
        while True:
            xmin = np.random.randint(0, img_w, 1)[0]
            ymin = np.random.randint(0, img_h, 1)[0]

            if xmin + w >= img_w or ymin + h >= img_h:
                continue

            tb_xmin = xmin / 32
            tb_xmax = (xmin + w) / 32
            tb_ymin = ymin / 32
            tb_ymax = (ymin + h) / 32

            if np.count_nonzero(np.r_[tb_mask[tb_ymin, tb_xmin], \
                                      tb_mask[tb_ymin, tb_xmax], \
                                      tb_mask[tb_ymax, tb_xmin], \
                                      tb_mask[tb_ymax, tb_xmax]]) >= 3:
                break
        regions.append((xmin, ymin, w, h))

    return regions

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

# def generate_suffix(train_sample_scheme=None, global_transform_scheme=None, local_transform_scheme=None):

#     suffix = []
#     if train_sample_scheme is not None:
#         suffix.append('trainSampleScheme_%d'%train_sample_scheme)
#     if global_transform_scheme is not None:
#         suffix.append('globalTxScheme_%d'%global_transform_scheme)
#     if local_transform_scheme is not None:
#         suffix.append('localTxScheme_%d'%local_transform_scheme)

#     return '_'.join(suffix)


class DataManager(object):

    ##########################
    ###    Annotation    #####
    ##########################

    @staticmethod
    def get_structure_pose_corrections(stack, stack_m=None,
                                classifier_setting_m=None,
                                classifier_setting_f=None,
                                warp_setting=None, trial_idx=None):
        basename = DataManager.get_warped_volume_basename(**locals())
        fp = os.path.join(ANNOTATION_ROOTDIR, stack, basename + '_' + 'structure3d_corrections' + '.pkl')
        return fp

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
    def load_annotation_to_grid_indices_lookup(stack, by_human, stack_m=None,
                                classifier_setting_m=None,
                                classifier_setting_f=None,
                                warp_setting=None, trial_idx=None):

        grid_indices_lookup_fp = DataManager.get_annotation_to_grid_indices_lookup_filepath(**locals())
        download_from_s3(grid_indices_lookup_fp)

        if os.path.exists(grid_indices_lookup_fp):
            raise Exception("Do not find structure to grid indices lookup file. Please generate it using `generate_annotation_to_grid_indices_lookup`")
        else:
            grid_indices_lookup = read_hdf(grid_indices_lookup_fp, 'grid_indices')
            return grid_indices_lookup

    @staticmethod
    def get_annotation_to_grid_indices_lookup_filepath(stack, by_human, stack_m=None,
                                classifier_setting_m=None,
                                classifier_setting_f=None,
                                warp_setting=None, trial_idx=None):
        if by_human:
            fp = os.path.join(ANNOTATION_ROOTDIR, stack, '%(stack)s_annotation_v3_grid_indices_lookup.hdf' % {'stack':stack})
        else:
            basename = DataManager.get_warped_volume_basename(stack_m=stack_m, stack_f=stack,
                                                              classifier_setting_m=classifier_setting_m,
                                                              classifier_setting_f=classifier_setting_f,
                                                              warp_setting=warp_setting, trial_idx=trial_idx)
            fp = os.path.join(ANNOTATION_ROOTDIR, stack, 'annotation_%(basename)s_grid_indices_lookup.hdf' % {'basename': basename})
        return fp

    @staticmethod
    def get_annotation_filepath(stack, by_human, stack_m=None,
                                classifier_setting_m=None,
                                classifier_setting_f=None,
                                warp_setting=None, trial_idx=None, suffix=None, timestamp=None):
        if by_human:
            fp = os.path.join(ANNOTATION_ROOTDIR, stack, '%(stack)s_annotation_v3.h5' % {'stack':stack})
        else:
            basename = DataManager.get_warped_volume_basename(stack_m=stack_m, stack_f=stack,
                                                              classifier_setting_m=classifier_setting_m,
                                                              classifier_setting_f=classifier_setting_f,
                                                              warp_setting=warp_setting, trial_idx=trial_idx)
            if suffix is not None:
                if timestamp is not None:
                    fp = os.path.join(ANNOTATION_ROOTDIR, stack, 'annotation_%(basename)s_%(suffix)s_%(timestamp)s.hdf' % {'basename': basename, 'suffix': suffix, 'timestamp': timestamp})
                else:
                    fp = os.path.join(ANNOTATION_ROOTDIR, stack, 'annotation_%(basename)s_%(suffix)s.hdf' % {'basename': basename, 'suffix': suffix})
            else:
                fp = os.path.join(ANNOTATION_ROOTDIR, stack, 'annotation_%(basename)s.hdf' % {'basename': basename})
        return fp

    @staticmethod
    def load_annotation_v3(stack=None, by_human=True, stack_m=None,
                                classifier_setting_m=None,
                                classifier_setting_f=None,
                                warp_setting=None, trial_idx=None, timestamp=None, suffix=None):
        if by_human:
            # fp = DataManager.get_annotation_filepath(stack, by_human=True)
            # download_from_s3(fp)
            # contour_df = DataManager.load_data(fp, filetype='annotation_hdf')
            #
            # try:
            #     structure_df = read_hdf(fp, 'structures')
            # except Exception as e:
            #     print e
            #     sys.stderr.write('Annotation has no structures.\n')
            #     return contour_df, None
            #
            # sys.stderr.write('Loaded annotation %s.\n' % fp)
            # return contour_df, structure_df
            raise Exception('Not implemented.')
        else:
            fp = DataManager.get_annotation_filepath(stack, by_human=False,
                                                     stack_m=stack_m,
                                                      classifier_setting_m=classifier_setting_m,
                                                      classifier_setting_f=classifier_setting_f,
                                                      warp_setting=warp_setting, trial_idx=trial_idx,
                                                    suffix=suffix, timestamp=timestamp)
            download_from_s3(fp)
            annotation_df = load_hdf_v2(fp)
            return annotation_df

    @staticmethod
    def get_annotation_viz_dir(stack):
        return os.path.join(ANNOTATION_VIZ_ROOTDIR, stack)

    @staticmethod
    def get_annotation_viz_filepath(stack, sec=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][sec]
        return os.path.join(ANNOTATION_VIZ_ROOTDIR, stack, fn + '_annotation_viz.tif')


    ########################################################

    @staticmethod
    def load_data(filepath, filetype):

        if not os.path.exists(filepath):
            sys.stderr.write('File does not exist: %s\n' % filepath)

        if filetype == 'bp':
            return bp.unpack_ndarray_file(filepath)
        elif filetype == 'image':
            return imread(filepath)
        elif filetype == 'hdf':
            try:
                return load_hdf(filepath)
            except:
                return load_hdf_v2(filepath)
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
    def get_anchor_filename_filename(stack):
        fn = THUMBNAIL_DATA_DIR + '/%(stack)s/%(stack)s_anchor.txt' % dict(stack=stack)
        return fn

    @staticmethod
    def load_anchor_filename(stack):
        fp = DataManager.get_anchor_filename_filename(stack)
        download_from_s3(fp, local_root=DATA_ROOTDIR)
        anchor_fn = DataManager.load_data(fp, filetype='anchor')
        return anchor_fn

    @staticmethod
    def get_cropbox_filename(stack, anchor_fn=None):
        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack=stack)
        fn = THUMBNAIL_DATA_DIR + '/%(stack)s/%(stack)s' % dict(stack=stack) + '_alignedTo_' + anchor_fn + '_cropbox.txt'
        return fn

    @staticmethod
    def load_cropbox(stack, anchor_fn=None):
        fp = DataManager.get_cropbox_filename(stack=stack, anchor_fn=anchor_fn)
        download_from_s3(fp, local_root=DATA_ROOTDIR)
        cropbox = DataManager.load_data(fp, filetype='bbox')
        return cropbox

    @staticmethod
    def get_sorted_filenames_filename(stack):
        fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_sorted_filenames.txt')
        return fn

    @staticmethod
    def load_sorted_filenames(stack):
        """
        Get the mapping between section index and image filename.

        Returns:
            Two dicts: filename_to_section, section_to_filename
        """

        fp = DataManager.get_sorted_filenames_filename(stack)
        download_from_s3(fp, local_root=DATA_ROOTDIR)
        filename_to_section, section_to_filename = DataManager.load_data(fp, filetype='file_section_map')
        if 'Placeholder' in filename_to_section:
            filename_to_section.pop('Placeholder')
        return filename_to_section, section_to_filename

    @staticmethod
    def get_transforms_filename(stack, anchor_fn=None):
        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]
        fn = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_transformsTo_%s.pkl' % anchor_fn)
        return fn

    @staticmethod
    def load_transforms(stack, downsample_factor, use_inverse, anchor_fn=None):
        """
        Args:
            use_inverse (bool): If True, load the transforms that when multiplied to a point on original space converts it to on aligned space. In preprocessing, set to False, which means simply parse the transform files as they are.

        """

        fp = DataManager.get_transforms_filename(stack, anchor_fn=anchor_fn)
        download_from_s3(fp)
        Ts = DataManager.load_data(fp, filetype='pickle')

        if use_inverse:
            Ts_inv_downsampled = {}
            for fn, T0 in Ts.iteritems():
                T = T0.copy()
                T[:2, 2] = T[:2, 2] * 32 / downsample_factor
                Tinv = np.linalg.inv(T)
                Ts_inv_downsampled[fn] = Tinv
            return Ts_inv_downsampled
        else:
            Ts_downsampled = {}
            for fn, T0 in Ts.iteritems():
                T = T0.copy()
                T[:2, 2] = T[:2, 2] * 32 / downsample_factor
                Ts_downsampled[fn] = T
            return Ts_downsampled

    #####################
    ### Registration ####
    #####################

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
    downscale=32, trial_idx=None):
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
        download_from_s3(params_fp)
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
    downscale=32, trial_idx=None):
        basename = DataManager.get_warped_volume_basename(**locals())

        if param_suffix is None:
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m,
                    basename, basename + '_scoreHistory.bp' % \
                    {'param_suffix':param_suffix})
        else:
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m,
                                basename, basename + '_scoreHistory_%(param_suffix)s.bp' % \
                                {'param_suffix':param_suffix})

    ####### Best trial index file #########

    @staticmethod
    def get_best_trial_index_filepath(stack_f, stack_m, warp_setting,
    classifier_setting_m=None, classifier_setting_f=None,
    type_f='score', type_m='score', downscale=32, param_suffix=None):
        basename = DataManager.get_warped_volume_basename(**locals())
        if param_suffix is None:
            fp = os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_bestTrial', basename + '_bestTrial.txt')
        else:
            fp = os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_bestTrial', basename + '_bestTrial_%(param_suffix)s.txt' % \
                             {'param_suffix':param_suffix})
        return fp

    @staticmethod
    def load_best_trial_index(stack_f, stack_m, warp_setting,
    classifier_setting_m=None, classifier_setting_f=None,
    type_f='score', type_m='score', downscale=32, param_suffix=None):
        fp = DataManager.get_best_trial_index_filepath(**locals())
        download_from_s3(fp)
        with open(fp, 'r') as f:
            best_trial_index = int(f.readline())
        return best_trial_index

    @staticmethod
    def load_best_trial_index_all_structures(stack_f, stack_m, warp_setting,
    classifier_setting_m=None, classifier_setting_f=None,
    type_f='score', type_m='score', downscale=32):
        input_kwargs = locals()
        best_trials = {}
        for structure in all_known_structures_sided:
            try:
                best_trials[structure] = DataManager.load_best_trial_index(param_suffix=structure, **input_kwargs)
            except Exception as e:
                sys.stderr.write(str(e) + '\n')
                sys.stderr.write("Best trial file for structure %s is not found.\n" % structure)
        return best_trials


    @staticmethod
    def get_alignment_viz_filepath(stack_m, stack_f,
                                classifier_setting_m,
                                classifier_setting_f,
                                warp_setting,
                                section,
                                type_m='score', type_f='score',
                                downscale=32,
                                trial_idx=None):

        basename = DataManager.get_warped_volume_basename(**locals())
        return os.path.join(REGISTRATION_VIZ_ROOTDIR, stack_m, basename, basename + '_%04d.jpg' % section)

    @staticmethod
    def load_confidence(stack_m, stack_f,
                            classifier_setting_m, classifier_setting_f, warp_setting, what,
                            type_m='score', type_f='score', param_suffix=None,
                            trial_idx=None):
        fp = DataManager.get_confidence_filepath(**locals())
        download_from_s3(fp)
        return load_pickle(fp)

    @staticmethod
    def get_confidence_filepath(stack_m, stack_f,
                            classifier_setting_m, classifier_setting_f, warp_setting, what,
                            type_m='score', type_f='score', param_suffix=None,
                            trial_idx=None):
        basename = DataManager.get_warped_volume_basename(**locals())

        if param_suffix is None:
            fn = basename + '_parameters' % {'param_suffix':param_suffix}
        else:
            fn = basename + '_parameters_%(param_suffix)s' % {'param_suffix':param_suffix}

        if what == 'hessians':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_hessians', fn + '_hessians.pkl')
        elif what == 'zscores':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_zscores', fn + '_zscores.pkl')
        elif what == 'score_landscape':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_scoreLandscape', fn + '_scoreLandscape.png')
        elif what == 'peak_width':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_peakWidth', fn + '_peakWidth.pkl')
        elif what == 'peak_radius':
            return os.path.join(REGISTRATION_PARAMETERS_ROOTDIR, stack_m, basename + '_peakRadius', fn + '_peakRadius.pkl')

        raise

    @staticmethod
    def get_classifier_filepath(structure, classifier_id):
        clf_fp = os.path.join(CLF_ROOTDIR, 'setting_%(setting)s', 'classifiers', '%(structure)s_clf_setting_%(setting)d.dump') % {'structure': structure, 'setting':classifier_id}
        return clf_fp

    @staticmethod
    def load_classifiers(classifier_id, structures=all_known_structures):

        from sklearn.externals import joblib

        clf_allClasses = {}
        for structure in structures:
            clf_fp = DataManager.get_classifier_filepath(structure=structure, classifier_id=classifier_id)
            download_from_s3(clf_fp)
            if os.path.exists(clf_fp):
                clf_allClasses[structure] = joblib.load(clf_fp)
            else:
                sys.stderr.write('Setting %d: No classifier found for %s.\n' % (classifier_id, structure))

        return clf_allClasses

    @staticmethod
    def load_sparse_scores(stack, structure, classifier_id, sec=None, fn=None, anchor_fn=None):

        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        sparse_scores_fn = DataManager.get_sparse_scores_filepath(stack=stack, structure=structure,
                                            classifier_id=classifier_id, fn=fn, anchor_fn=anchor_fn)
        download_from_s3(sparse_scores_fn)
        return DataManager.load_data(sparse_scores_fn, filetype='bp')

    @staticmethod
    def get_sparse_scores_filepath(stack, structure, classifier_id, sec=None, fn=None, anchor_fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        return os.path.join(SPARSE_SCORES_ROOTDIR, stack, '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped', \
                '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_sparseScores_setting_%(classifier_id)s.hdf') % \
                {'fn': fn, 'anchor_fn': anchor_fn, 'structure':structure, 'classifier_id': classifier_id}

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

    ################
    # Mesh related #
    ################

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
                                    trial_idx=None,
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
                                    trial_idx=None,
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
                                            trial_idx=None, **kwargs):
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
                                            trial_idx=None, **kwargs):
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
                                        trial_idx=None,
                                        sided=True,
                                        include_surround=False):
        if stack_f is not None:
            return DataManager.load_transformed_volume_all_known_structures(**locals())
        else:
            raise Exception('Not implemented.')


    @staticmethod
    def load_transformed_volume(stack_m, stack_f,
                                        warp_setting,
                                        classifier_setting_m=None,
                                        classifier_setting_f=None,
                                        type_m='score',
                                         type_f='score',
                                        structure=None,
                                        downscale=32,
                                        trial_idx=None):
        fp = DataManager.get_transformed_volume_filepath(**locals())
        download_from_s3(fp)
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
                                                    sided=True,
                                                    include_surround=False,
                                                     trial_idx=None,
):
        """
        Args:
            trial_idx: could be int (for global transform) or dict {sided structure name: best trial index} (for local transform).
        """

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
                if trial_idx is None or isinstance(trial_idx, int):
                    volumes[structure] = DataManager.load_transformed_volume(stack_m=stack_m, type_m=type_m,
                                                        stack_f=stack_f, type_f=type_f, downscale=downscale,
                                                        classifier_setting_m=classifier_setting_m,
                                                        classifier_setting_f=classifier_setting_f,
                                                        warp_setting=warp_setting,
                                                        structure=structure,
                                                        trial_idx=trial_idx)
                else:
                    volumes[structure] = DataManager.load_transformed_volume(stack_m=stack_m, type_m=type_m,
                                                        stack_f=stack_f, type_f=type_f, downscale=downscale,
                                                        classifier_setting_m=classifier_setting_m,
                                                        classifier_setting_f=classifier_setting_f,
                                                        warp_setting=warp_setting,
                                                        structure=structure,
                                                        trial_idx=trial_idx[convert_to_nonsurround_label(structure)])
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
                                        trial_idx=None):

        basename = DataManager.get_warped_volume_basename(**locals())
        if structure is not None:
            fn = basename + '_%s' % structure
        return os.path.join(VOLUME_ROOTDIR, stack_m, basename, 'score_volumes', fn + '.bp')


    ##########################
    ## Probabilistic Shape  ##
    ##########################

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
                                        trial_idx=None):

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
        download_from_s3(vol_fp, is_dir=False)
        volume = DataManager.load_data(vol_fp, filetype='bp')
        return volume

    @staticmethod
    def load_original_volume_bbox(stack, vol_type, classifier_setting=None, structure=None, downscale=32):
        """
        This returns the 3D bounding box.

        Args:
            vol_type (str):
                annotation: with respect to aligned uncropped thumbnail
                score/thumbnail: with respect to aligned cropped thumbnail
                shell: with respect to aligned uncropped thumbnail
        """
        bbox_fp = DataManager.get_original_volume_bbox_filepath(**locals())
        download_from_s3(bbox_fp)
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

    #########################
    ###     Score map     ###
    #########################

    @staticmethod
    def get_scoremap_viz_filepath(stack, downscale, section=None, fn=None, anchor_fn=None, structure=None, classifier_id=None):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn): raise Exception('Section is invalid: %s.' % fn)

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        viz_dir = os.path.join(ROOT_DIR, 'CSHL_scoremaps_down%(down)d_viz' % dict(down=downscale))
        basename = '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_down%(down)d' % dict(fn=fn, anchor_fn=anchor_fn, down=downscale)
        fn = basename + '_%(structure)s_denseScoreMap_setting_%(classifier_id)d.jpg' % dict(structure=structure, classifier_id=classifier_id)

        scoremap_viz_filepath = os.path.join(viz_dir, structure, stack, fn)

        return scoremap_viz_filepath

    @staticmethod
    def get_downscaled_scoremap_filepath(stack, structure, classifier_id, downscale, section=None, fn=None, anchor_fn=None, return_bbox_fp=False):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn):
                raise Exception('Section is invalid: %s.' % fn)

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        basename = '%(fn)s_alignedTo_%(anchor_fn)s_cropped_down%(down)d' % dict(fn=fn, anchor_fn=anchor_fn, down=downscale)

        scoremap_bp_filepath = os.path.join(ROOT_DIR, 'CSHL_scoremaps_down%(down)d' % dict(down=downscale), stack, basename,
        basename + '_%(structure)s_denseScoreMap_setting_%(classifier_id)d.bp' % dict(structure=structure, classifier_id=classifier_id))

        return scoremap_bp_filepath

    @staticmethod
    def load_downscaled_scoremap(stack, structure, classifier_id, section=None, fn=None, anchor_fn=None, downscale=32):
        """
        Return scoremaps as bp files.
        """

        # Load scoremap
        scoremap_bp_filepath = DataManager.get_downscaled_scoremap_filepath(stack, section=section, \
                        fn=fn, anchor_fn=anchor_fn, structure=structure, classifier_id=classifier_id,
                        downscale=downscale)

        download_from_s3(scoremap_bp_filepath)

        if not os.path.exists(scoremap_bp_filepath):
            raise Exception('No scoremap for image %s (section %d) for label %s\n' % \
            (metadata_cache['sections_to_filenames'][stack][section], section, structure))

        scoremap_downscaled = DataManager.load_data(scoremap_bp_filepath, filetype='bp')
        return scoremap_downscaled

    @staticmethod
    def get_scoremap_filepath(stack, structure, classifier_id, section=None, fn=None, anchor_fn=None, return_bbox_fp=False):

        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn):
                raise Exception('Section is invalid: %s.' % fn)

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        scoremap_bp_filepath = os.path.join(SCOREMAPS_ROOTDIR, stack, \
        '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_denseScoreMap_setting_%(classifier_id)d.hdf') \
        % dict(stack=stack, fn=fn, structure=structure, anchor_fn=anchor_fn, classifier_id=classifier_id)

        scoremap_bbox_filepath = os.path.join(SCOREMAPS_ROOTDIR, stack, \
        '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_%(structure)s_denseScoreMap_interpBox.txt') \
            % dict(stack=stack, fn=fn, structure=structure, anchor_fn=anchor_fn)

        if return_bbox_fp:
            return scoremap_bp_filepath, scoremap_bbox_filepath
        else:
            return scoremap_bp_filepath

    @staticmethod
    def load_scoremap(stack, structure, classifier_id, section=None, fn=None, anchor_fn=None, downscale=1):
        """
        Return scoremaps.
        """

        # Load scoremap
        scoremap_bp_filepath, scoremap_bbox_filepath = DataManager.get_scoremap_filepath(stack, section=section, \
                                    fn=fn, anchor_fn=anchor_fn, structure=structure, return_bbox_fp=True, classifier_id=classifier_id)
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

    ###########################
    ######  CNN Features ######
    ###########################

    @staticmethod
    def load_dnn_feature_locations(stack, model_name, section=None, fn=None, anchor_fn=None, input_img_version='cropped_gray'):
        fp = DataManager.get_dnn_feature_locations_filepath(stack=stack, model_name=model_name, section=section, fn=fn, anchor_fn=anchor_fn, input_img_version=input_img_version)
        download_from_s3(fp)
        locs = np.loadtxt(fp).astype(np.int)
        indices = locs[:, 0]
        locations = locs[:, 1:]
        return indices, locations

    @staticmethod
    def get_dnn_feature_locations_filepath(stack, model_name, section=None, fn=None, anchor_fn=None, input_img_version='cropped_gray'):

        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]
            
        image_version_basename = DataManager.get_image_version_basename(stack=stack, resol='lossless', version=input_img_version)
        image_basename = DataManager.get_image_basename(stack=stack, fn=fn, resol='lossless', version=input_img_version)

        # feature_locs_fn = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack, \
        # '%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped/%(fn)s_lossless_alignedTo_%(anchor_fn)s_cropped_patch_locations.txt' % \
        # dict(fn=fn, anchor_fn=anchor_fn))
        
        feature_locs_fn = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack, image_version_basename,
                                       image_basename + '_patch_locations.txt')
        return feature_locs_fn

    @staticmethod
    def get_dnn_features_filepath(stack, model_name, section=None, fn=None, anchor_fn=None, input_img_version='cropped_gray'):
        """
        Args:
            version (str): default is cropped_gray.
        """

        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]

        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]

        image_version_basename = DataManager.get_image_version_basename(stack=stack, resol='lossless', version=input_img_version)
        image_basename = DataManager.get_image_basename(stack=stack, fn=fn, resol='lossless', version=input_img_version)

        feature_fn = os.path.join(PATCH_FEATURES_ROOTDIR, model_name, stack, image_version_basename, image_basename + '_features.bp')

        return feature_fn

    @staticmethod
    def load_dnn_features(stack, model_name, section=None, fn=None, anchor_fn=None, input_img_version='cropped_gray'):
        """
        Args:
            version (str): default is cropped_gray.
        """
        
        features_fp = DataManager.get_dnn_features_filepath(stack=stack, model_name=model_name, section=section, fn=fn, anchor_fn=anchor_fn, input_img_version=input_img_version)
        download_from_s3(features_fp)

        try:
            return load_hdf(features_fp)
        except:
            pass

        try:
            return load_hdf_v2(features_fp)
        except:
            pass

        return bp.unpack_ndarray_file(features_fp)

    ##################
    ##### Image ######
    ##################
    
    @staticmethod
    def get_image_version_basename(stack, version, resol='lossless', anchor_fn=None):
        
        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]
        
        if resol == 'lossless' and (version == 'cropped' or version == 'cropped_tif'):
            image_version_basename = stack + '_' + resol + '_alignedTo_' + anchor_fn + '_cropped'
        else:
            image_version_basename = stack + '_' + resol + '_alignedTo_' + anchor_fn + '_' + version
        
        return image_version_basename
    
    @staticmethod
    def get_image_basename(stack, version, resol='lossless', anchor_fn=None, fn=None, section=None):
        
        if anchor_fn is None:
            anchor_fn = metadata_cache['anchor_fn'][stack]
            
        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            assert is_invalid(fn=fn), 'Section is invalid: %s.' % fn
        
        if resol == 'lossless' and (version == 'cropped' or version == 'cropped_tif'):
            image_basename = fn + '_' + resol + '_alignedTo_' + anchor_fn + '_cropped'
        else:
            image_basename = fn + '_' + resol + '_alignedTo_' + anchor_fn + '_' + version
        
        return image_basename

    @staticmethod
    def get_image_dir(stack, version, resol='lossless', anchor_fn=None, modality=None,
                      data_dir=DATA_DIR, raw_data_dir=RAW_DATA_DIR, thumbnail_data_dir=THUMBNAIL_DATA_DIR):
        """
        Args:
            data_dir: This by default is DATA_DIR, but one can change this ad-hoc when calling the function
            resol: can be either lossless or thumbnail
            version: TODO - Write a list of options
            modality: can be either nissl or fluorescent. If not specified, it is inferred.

        Returns:
            Absolute path of the image directory.
        """

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack)

        if resol == 'lossless' and version == 'original_jp2':
            image_dir = os.path.join(raw_data_dir, stack)
        elif resol == 'lossless' and version == 'compressed':
            image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_compressed' % {'anchor_fn':anchor_fn})
        elif resol == 'lossless' and version == 'uncropped_tif':
            image_dir = os.path.join(data_dir, stack, stack + '_' + resol + '_tif')
        elif resol == 'lossless' and version == 'cropped_16bit':
            image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
        # elif resol == 'lossless' and version == 'cropped_gray_contrast_stretched':
        #     image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_gray_contrast_stretched' % {'anchor_fn':anchor_fn})
        # elif resol == 'lossless' and version == 'cropped_8bit_blueasgray':
        #     image_dir = os.path.join(data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped_contrast_stretched_blueasgray' % {'anchor_fn':anchor_fn})
        elif resol == 'thumbnail' and (version == 'cropped' or version == 'cropped_tif'):
            image_dir = os.path.join(thumbnail_data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn})
        elif (resol == 'thumbnail' and version == 'aligned') or (resol == 'thumbnail' and version == 'aligned_tif'):
            image_dir = os.path.join(thumbnail_data_dir, stack, stack+'_'+resol+'_alignedTo_%(anchor_fn)s' % {'anchor_fn':anchor_fn})
        elif resol == 'thumbnail' and version == 'original_png':
            image_dir = os.path.join(raw_data_dir, stack)
        else:
            # sys.stderr.write('No special rule for (%s, %s). So using the default image directory composition rule.\n' % (version, resol))
            image_dir = os.path.join(data_dir, stack, stack + '_' + resol + '_alignedTo_' + anchor_fn + '_' + version)

        return image_dir

    @staticmethod
    def load_image(stack, version, resol='lossless', section=None, fn=None, anchor_fn=None, modality=None, data_dir=DATA_DIR, ext=None):
        img_fp = DataManager.get_image_filepath(**locals())
        download_from_s3(img_fp)
        return imread(img_fp)

    @staticmethod
    def get_image_filepath(stack, version, resol='lossless',
                           data_dir=DATA_DIR, raw_data_dir=RAW_DATA_DIR, thumbnail_data_dir=THUMBNAIL_DATA_DIR,
                           section=None, fn=None, anchor_fn=None, modality=None, ext=None):
        """
        Args:
            data_dir: This by default is DATA_DIR, but one can change this ad-hoc when calling the function
            resol: can be either lossless or thumbnail
            version: TODO - Write a list of options
            modality: can be either nissl or fluorescent. If not specified, it is inferred.

        Returns:
            Absolute path of the image file.
        """

        image_name = None
        
        if section is not None:
            fn = metadata_cache['sections_to_filenames'][stack][section]
            if is_invalid(fn=fn):
                raise Exception('Section is invalid: %s.' % fn)
        else:
            assert fn is not None

        if anchor_fn is None:
            anchor_fn = DataManager.load_anchor_filename(stack)

        if modality is None:
            if (stack in all_alt_nissl_ntb_stacks or stack in all_alt_nissl_tracing_stacks) and fn.split('-')[1][0] == 'F':
                modality = 'fluorescent'
            else:
                modality = 'nissl'

        image_dir = DataManager.get_image_dir(stack=stack, version=version, resol=resol, modality=modality, data_dir=data_dir)

        if resol == 'thumbnail' and version == 'original_png':
            image_name = fn + '.png'
        # elif resol == 'lossless' and (version == 'compressed' or version == 'contrast_stretched_compressed' or version == 'cropped_compressed'):
        #     if modality == 'fluorescent':
        #         image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_contrast_stretched_compressed.jpg' % {'anchor_fn':anchor_fn}])
        #     else:
        #         image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_compressed.jpg' % {'anchor_fn':anchor_fn}])
        elif resol == 'lossless' and version == 'cropped':
            ext = 'tif'
            # if modality == 'fluorescent':
            #     image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_contrast_stretched' % {'anchor_fn':anchor_fn}])
            # else:
            #     image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped' % {'anchor_fn':anchor_fn}])
            image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped.tif' % {'anchor_fn':anchor_fn}])
        elif resol == 'lossless' and version == 'uncropped_tif':
            image_name = fn + '_lossless.tif'
        # elif resol == 'lossless' and version == 'cropped_gray_contrast_stretched':
            # image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_gray_contrast_stretched.tif' % {'anchor_fn':anchor_fn}])
        # elif resol == 'lossless' and version == 'cropped_16bit':
            # image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped.tif' % {'anchor_fn':anchor_fn}])            
        # elif resol == 'lossless' and version == 'cropped_gray':
        #     ext = 'tif'
        #     image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_gray.tif' % {'anchor_fn':anchor_fn}])
        # elif resol == 'lossless' and version == 'cropped_gray_jpeg':
        #     image_name = fn + '_' + resol + '_alignedTo_' + anchor_fn + '_cropped_gray.jpg'            
        # elif resol == 'lossless' and version == 'cropped_gray_linearNormalized':
        #     image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_gray_linearNormalized.tif' % {'anchor_fn':anchor_fn}])
        # elif resol == 'lossless' and version == 'cropped_gray_linearNormalized_jpeg':
        #     image_name = '_'.join([fn, resol, 'alignedTo_%(anchor_fn)s_cropped_gray_linearNormalized.jpg' % {'anchor_fn':anchor_fn}])
        elif resol == 'thumbnail' and (version == 'cropped' or version == 'cropped_tif'):
            image_name = '_'.join([fn, resol, 'alignedTo_' + anchor_fn + '_cropped.tif'])
        elif resol == 'thumbnail' and (version == 'aligned' or version == 'aligned_tif'):
            image_name = '_'.join([fn, resol, 'alignedTo_' + anchor_fn + '.tif'])
        #else:
            # sys.stderr.write('No special rule for (%s, %s). So using the default image filepath composition rule.\n' % (version, resol))
        
        if image_name is None:
            if ext is None:
                ext = 'tif'
            image_name = '_'.join([fn, resol, 'alignedTo_' + anchor_fn + '_' + version + '.' + ext])    
            
        image_path = os.path.join(image_dir, image_name)
            
        return image_path


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
            fp = DataManager.get_image_filepath(stack=stack, resol='lossless', version='cropped', fn=random_fn, anchor_fn=anchor_fn)
            download_from_s3(fp)
            if not os.path.exists(fp):
                continue
            image_width, image_height = map(int, check_output("identify -format %%Wx%%H %s" % fp, shell=True).split('x'))
            break

        return image_width, image_height

    #######################################################

    @staticmethod
    def convert_section_to_z(stack, sec, downsample, z_begin=None, first_sec=None):
        """
        Because the z-spacing is much larger than pixel size on x-y plane,
        the theoretical voxels are square on x-y plane but elongated in z-direction.
        In practice we need to represent volume using cubic voxels.
        This function computes the z-coordinate for a given section number.
        This depends on the downsample factor of the volume.

        Args:
            downsample (int):
            first_sec (int): default to the first brainstem section defined in ``cropbox".
            z_begin (float): default to the z position of the first_sec.

        Returns:
            z1, z2 (2-tuple of float): in
        """

        xy_pixel_distance = XY_PIXEL_DISTANCE_LOSSLESS * downsample
        voxel_z_size = SECTION_THICKNESS / xy_pixel_distance
        # Voxel size in z direction in unit of x,y pixel.

        if first_sec is None:
            first_sec, _ = DataManager.load_cropbox(stack)[4:]

        if z_begin is None:
            z_begin = first_sec * voxel_z_size

        z1 = sec * voxel_z_size
        z2 = (sec + 1) * voxel_z_size
        return z1-z_begin, z2-1-z_begin

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

    ############################
    #####    Masks     #########
    ############################


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
    def get_thumbnail_mask_dir_v3(stack, version='aligned'):
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
    def get_thumbnail_mask_filename_v3(stack, section=None, fn=None, version='aligned_cropped'):
        """
        Get filepath of thumbnail mask.

        Args:
            version (str): One of aligned, aligned_cropped, cropped.
        """

        anchor_fn = metadata_cache['anchor_fn'][stack]
        dir_path = DataManager.get_thumbnail_mask_dir_v3(stack, version=version)
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][section]

        if version == 'aligned':
            fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_mask.png')
        elif version == 'aligned_cropped' or version == 'cropped':
            fp = os.path.join(dir_path, fn + '_alignedTo_' + anchor_fn + '_mask_cropped.png')
        else:
            raise Exception('version %s is not recognized.' % version)
        return fp


    @staticmethod
    def load_thumbnail_mask_v3(stack, section=None, fn=None, version='aligned_cropped'):
        fp = DataManager.get_thumbnail_mask_filename_v3(stack=stack, section=section, fn=fn, version=version)
        download_from_s3(fp)
        mask = DataManager.load_data(fp, filetype='image').astype(np.bool)
        return mask

    @staticmethod
    def load_thumbnail_mask_v2(stack, section=None, fn=None, version='aligned_cropped'):
        fp = DataManager.get_thumbnail_mask_filename_v2(stack=stack, section=section, fn=fn, version=version)
        download_from_s3(fp, local_root=DATA_ROOTDIR)
        mask = DataManager.load_data(fp, filetype='image').astype(np.bool)
        return mask

    @staticmethod
    def get_thumbnail_mask_dir_v2(stack, version='aligned_cropped'):
        anchor_fn = metadata_cache['anchor_fn'][stack]
        if version == 'aligned_cropped':
            mask_dir = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_masks_alignedTo_' + anchor_fn + '_cropped')
        elif version == 'aligned':
            mask_dir = os.path.join(THUMBNAIL_DATA_DIR, stack, stack + '_masks_alignedTo_' + anchor_fn)
        else:
            raise Exception("version %s not recognized." % version)
        return mask_dir

    @staticmethod
    def get_thumbnail_mask_filename_v2(stack, section=None, fn=None, version='aligned_cropped'):
        anchor_fn = metadata_cache['anchor_fn'][stack]
        sections_to_filenames = metadata_cache['sections_to_filenames'][stack]
        if fn is None:
            fn = sections_to_filenames[section]
        mask_dir = DataManager.get_thumbnail_mask_dir_v2(stack=stack, version=version)
        if version == 'aligned_cropped':
            fp = os.path.join(mask_dir, fn + '_mask_alignedTo_' + anchor_fn + '_cropped.png')
        elif version == 'aligned':
            fp = os.path.join(mask_dir, fn + '_mask_alignedTo_' + anchor_fn + '.png')
        else:
            raise Exception("version %s not recognized." % version)
        return fp

    ###################################

    @staticmethod
    def get_region_labels_filepath(stack, sec=None, fn=None):
        """
        Returns:
            dict {label: list of region indices}
        """
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        return os.path.join(CELL_FEATURES_CLF_ROOTDIR, 'region_indices_by_label', stack, fn + '_region_indices_by_label.hdf')

    @staticmethod
    def get_ntb_to_nissl_intensity_profile_mapping_filepath(stack=None, ntb_fn=None):
        """
        Args:
            stack (str): If None, read the a priori mapping.
        """
        if stack is None:
            fp = os.path.join(DATA_DIR, 'average_nissl_intensity_mapping.npy')
        else:
            fp = os.path.join(DATA_DIR, stack, stack + '_intensity_mapping', '%s_intensity_mapping.npy' % (ntb_fn))

        return fp

    @staticmethod
    def get_dataset_features_filepath(dataset_id):
        features_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_features.bp')
        return features_fp

    @staticmethod
    def get_dataset_addresses_filepath(dataset_id):
        addresses_fp = os.path.join(CLF_ROOTDIR, 'datasets', 'dataset_%d' % dataset_id, 'patch_addresses.pkl')
        return addresses_fp

    @staticmethod
    def get_classifier_filepath(classifier_id, structure):
        classifier_id_dir = os.path.join(CLF_ROOTDIR, 'setting_%d' % classifier_id)
        classifier_dir = os.path.join(classifier_id_dir, 'classifiers')
        return os.path.join(classifier_dir, '%(structure)s_clf_setting_%(setting)d.dump' % \
                     dict(structure=structure, setting=classifier_id))

    ####### Fluorescent ########

    @staticmethod
    def get_labeled_neurons_filepath(stack, sec=None, fn=None):
        if fn is None:
            fn = metadata_cache['sections_to_filenames'][stack][sec]
        return os.path.join(LABELED_NEURONS_ROOTDIR, stack, fn, fn + ".pkl")


    @staticmethod
    def load_labeled_neurons_filepath(stack, sec=None, fn=None):
        fp = DataManager.get_labeled_neurons_filepath(**locals())
        download_from_s3(fp)
        return load_pickle(fp)

##################################################

def download_all_metadata():

    # Download preprocess data
    for stack in all_stacks:
        download_from_s3(DataManager.get_sorted_filenames_filename(stack=stack))
        download_from_s3(DataManager.get_anchor_filename_filename(stack=stack))
        download_from_s3(DataManager.get_cropbox_filename(stack=stack))

download_all_metadata()

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
     'MD652': (22720, 15552),
     'MD653': (25664, 16800),
     'MD657': (27584, 16960),
     'MD658': (19936, 15744)}
    metadata_cache['anchor_fn'] = {}
    metadata_cache['sections_to_filenames'] = {}
    metadata_cache['filenames_to_sections'] = {}
    metadata_cache['section_limits'] = {}
    metadata_cache['cropbox'] = {}
    metadata_cache['valid_sections'] = {}
    metadata_cache['valid_filenames'] = {}
    metadata_cache['valid_sections_all'] = {}
    metadata_cache['valid_filenames_all'] = {}
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
            metadata_cache['filenames_to_sections'][stack] = DataManager.load_sorted_filenames(stack)[0]
            metadata_cache['filenames_to_sections'][stack].pop('Placeholder')
            metadata_cache['filenames_to_sections'][stack].pop('Nonexisting')
            metadata_cache['filenames_to_sections'][stack].pop('Rescan')
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

        try:
            first_sec, last_sec = metadata_cache['section_limits'][stack]
            metadata_cache['valid_sections'][stack] = [sec for sec in range(first_sec, last_sec+1) if not is_invalid(stack=stack, sec=sec)]
            metadata_cache['valid_filenames'][stack] = [metadata_cache['sections_to_filenames'][stack][sec] for sec in
                                                       metadata_cache['valid_sections'][stack]]
        except:
            pass
        
        try:
            metadata_cache['valid_sections_all'][stack] = [sec for sec, fn in metadata_cache['sections_to_filenames'][stack].iteritems() if not is_invalid(fn=fn)]
            metadata_cache['valid_filenames_all'][stack] = [fn for sec, fn in metadata_cache['sections_to_filenames'][stack].iteritems() if not is_invalid(fn=fn)]
        except:
            pass

generate_metadata_cache()

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
