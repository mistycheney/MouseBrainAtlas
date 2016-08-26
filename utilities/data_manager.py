from utilities2015 import *
from metadata import *

class DataManager(object):


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
    def get_thumbnail_mask_filepath(stack, section, cerebellum_removed=False):
        if cerebellum_removed:
            fn = data_dir+'/%(stack)s_thumbnail_aligned_mask_cropped_cerebellumRemoved/%(stack)s_%(sec)04d_thumbnail_aligned_mask_cropped_cerebellumRemoved.png' % \
                {'stack': stack, 'sec': section}
        else:
            fn = data_dir+'/%(stack)s_thumbnail_aligned_mask_cropped/%(stack)s_%(sec)04d_thumbnail_aligned_mask_cropped.png' % \
                            {'stack': stack, 'sec': section}
        return fn


    @staticmethod
    def get_image_filepath(stack, section, version='rgb-jpg', resol='lossless', data_dir=data_dir):
        # if data_dir is None:
        #     data_dir = os.environ['DATA_DIR']

        # if resol is None:
        #     resol = 'lossless'

        slice_str = '%04d' % section

        if version == 'rgb-jpg':
            image_dir = os.path.join(data_dir, stack+'_'+resol+'_aligned_cropped_downscaled')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_downscaled'])
            image_path = os.path.join(image_dir, image_name + '.jpg')
        # elif version == 'gray-jpg':
        #     image_dir = os.path.join(self.data_dir, stack+'_'+resol+'_cropped_grayscale_downscaled')
        #     image_name = '_'.join([stack, slice_str, resol, 'warped'])
        #     image_path = os.path.join(image_dir, image_name + '.jpg')
        # elif version == 'gray':
        #     image_dir = os.path.join(data_dir, stack+'_'+resol+'_aligned_cropped_grayscale')
        #     image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_grayscale'])
        #     image_path = os.path.join(image_dir, image_name + '.tif')
        # elif version == 'rgb':
        #     image_dir = os.path.join(data_dir, stack+'_'+resol+'_aligned_cropped')
        #     image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped'])
        #     image_path = os.path.join(image_dir, image_name + '.tif')
        # elif version == 'stereotactic-rgb-jpg':
        #     image_dir = os.path.join(data_dir, stack+'_'+resol+'_aligned_cropped_downscaled_stereotactic')
        #     image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_downscaled_stereotactic'])
        #     image_path = os.path.join(image_dir, image_name + '.jpg')

        elif version == 'saturation':
            image_dir = os.path.join(data_dir, stack+'_'+resol+'_aligned_cropped_saturation')
            image_name = '_'.join([stack, slice_str, resol, 'aligned_cropped_saturation'])
            image_path = os.path.join(image_dir, image_name + '.jpg')

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
        try:
            sec = section_range_lookup[stack][0]
            image_width, image_height = map(int, check_output("identify -format %%Wx%%H %s" % DataManager.get_image_filepath(stack=stack, section=sec, version='rgb-jpg', data_dir=data_dir),
            shell=True).split('x'))
        except Exception as e:
            print e
            # sys.stderr.write('Cannot find image.\n')

        return image_width, image_height

    @staticmethod
    def convert_section_to_z(stack, sec, downsample, z_begin=None):
        """
        z_begin default to the first brainstem section.
        """

        xy_pixel_distance = xy_pixel_distance_lossless * downsample
        voxel_z_size = section_thickness / xy_pixel_distance
        # print 'voxel size:', xy_pixel_distance, xy_pixel_distance, voxel_z_size, 'um'

        first_sec, last_sec = section_range_lookup[stack]
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

        first_sec, last_sec = section_range_lookup[stack]
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
