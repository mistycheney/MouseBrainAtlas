#! /usr/bin/env python

import sys, os
import argparse
from sys import argv, exit

from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from ui.ui_MaskEditingGui4 import Ui_MaskEditingGui
from widgets.DrawableZoomableBrowsableGraphicsScene import DrawableZoomableBrowsableGraphicsScene
from widgets.DrawableZoomableBrowsableGraphicsScene_ForMasking import DrawableZoomableBrowsableGraphicsScene_ForMasking
from widgets.ZoomableBrowsableGraphicsScene import ZoomableBrowsableGraphicsScene
from DataFeeder import ImageDataFeeder

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *
from registration_utilities import find_contour_points
from gui_utilities import *
from qt_utilities import *
from mask_editing_utilities import *
sys.path.append(os.path.join(os.environ['REPO_DIR'], 'web_services'))
from web_service import web_services_request

STR_USING_AUTO = 'Using AUTO (Click to switch to USER)'
STR_USING_USER = 'Using USER (Click to switch to AUTO)'

class MaskEditingGUI(QMainWindow):
    def __init__(self, parent=None, stack=None):
        QMainWindow.__init__(self, parent)

        self.stack = stack

        self.ui = Ui_MaskEditingGui()
        self.dialog = QDialog(self)
        self.ui.setupUi(self.dialog)

        self.ui.button_slic.clicked.connect(self.update_slic)
        self.ui.button_submasks.clicked.connect(self.update_init_submasks_image)
        self.ui.button_snake.clicked.connect(self.update_user_submasks_image)
        self.ui.button_update_merged_mask.clicked.connect(self.update_merged_mask_clicked)
        self.ui.button_toggle_accept_auto.clicked.connect(self.toggle_accept_auto)
        self.ui.button_toggle_accept_auto.setText(STR_USING_AUTO)
        # self.ui.button_save.clicked.connect(self.save_current_section)
        self.ui.button_saveAll.clicked.connect(self.save_all)
        self.ui.button_uploadMasks.clicked.connect(self.upload_masks)
        # self.ui.autogenMasks.clicked.connect(self.generate_masks)
        # self.ui.button_confirmFinalMask.clicked.connect(self.confirm_final_masks)

        # self.ui.slider_threshold.setSingleStep(1)
        # self.ui.slider_threshold.setMinimum(1)
        # self.ui.slider_threshold.setMaximum(255)
        # self.ui.slider_threshold.setValue(200)
        # self.ui.slider_threshold.setEnabled(True)
        # self.ui.slider_threshold.valueChanged.connect(self.threshold_changed)

        self.ui.slider_dissimThresh.setSingleStep(1) # unit is 0.01
        self.ui.slider_dissimThresh.setMinimum(0)
        self.ui.slider_dissimThresh.setMaximum(200) # 2
        self.ui.slider_dissimThresh.setValue(30) # 0.3
        self.ui.slider_dissimThresh.valueChanged.connect(self.dissim_threshold_changed)
        # self.ui.button_confirmDissimThresh.clicked.connect(self.dissim_threshold_change_confirmed)

        self.ui.slider_snakeShrink.setSingleStep(1)
        self.ui.slider_snakeShrink.setMinimum(0)
        self.ui.slider_snakeShrink.setMaximum(20)
        self.ui.slider_snakeShrink.setValue(MORPHSNAKE_LAMBDA1)
        self.ui.slider_snakeShrink.valueChanged.connect(self.snake_shrinkParam_changed)

        self.sections_to_filenames = DataManager.load_sorted_filenames(stack)[1]
        self.valid_sections_to_filenames = {sec: fn for sec, fn in self.sections_to_filenames.iteritems() if not is_invalid(fn)}
        self.valid_filenames_to_sections = {fn: sec for sec, fn in self.valid_sections_to_filenames.iteritems()}
        q = sorted(self.valid_sections_to_filenames.items())
        self.valid_sections = [sec for sec, fn in q]
        self.valid_filenames = [fn for sec, fn in q]

        ########################################################

        self.original_images = {}
        self.selected_channels = {}
        # self.selected_thresholds = {sec: 200 for sec in self.valid_sections}
        self.thresholded_images = {}
        self.contrast_stretched_images = {}
        self.slic_labelmaps = {}
        self.slic_boundary_images = {}
        self.ncut_labelmaps = {}
        # self.border_dissim_images = {}
        self.selected_dissim_thresholds = {}
        self.selected_snake_lambda1 = {}
        self.sp_dissim_maps = {}
        self.init_submasks = {}
        self.init_submasks_vizs = {}
        self.user_submasks = {}
        # self.final_submasks_vizs = {}
        self.accepted_final_masks = {}
        self.accept_which = {sec: 0 for sec in self.valid_sections}
        self.merged_masks = {}
        self.merged_mask_vizs = {}

        self.auto_submask_decisions = {}
        self.user_submask_decisions = {}

        # Load decisions from final decision file.
        try:
            accept_which, submask_decisions = load_final_decisions(self.stack)
            for fn, which in accept_which.iteritems():
                if fn not in self.valid_filenames:
                    continue
                sec = self.valid_filenames_to_sections[fn]
                if which == 0:
                    self.auto_submask_decisions[sec] = submask_decisions[fn]
                elif which == 1:
                    self.user_submask_decisions[sec] = submask_decisions[fn]
                else:
                    raise Exception('The type indicator %s must be either 0 or 1, but is %d' % (fn, which))
                self.accept_which[sec] = which
            sys.stderr.write('Decisions loaded from the final decision file.\n')
        except Exception as e:
            sys.stderr.write('Error loading final decision file.\n')

        ######################################
        ## Generate submask review results. ##
        ######################################

        self.auto_submasks_gscene = DrawableZoomableBrowsableGraphicsScene_ForMasking(id='autoFinalMasks', gview=self.ui.gview_final_masks_auto)
        self.auto_masks_feeder = ImageDataFeeder(name='autoFinalMasks', stack=self.stack, \
                                    sections=self.valid_sections, use_data_manager=False, downscale=32,
                                    labeled_filenames={sec: os.path.join(RAW_DATA_DIR, self.stack, fn + ".png")
                                        for sec, fn in self.valid_sections_to_filenames.iteritems()})
        self.auto_submasks_gscene.set_data_feeder(self.auto_masks_feeder)
        self.auto_submasks_gscene.active_image_updated.connect(self.auto_submasks_gscene_section_changed)
        self.auto_submasks_gscene.submask_decision_updated.connect(self.auto_submask_decision_updated)

        submasks_rootdir = os.path.join(THUMBNAIL_DATA_DIR, self.stack, self.stack + '_submasks')

        def filter_by_keys(d, allowed_key_list):
            return {fn: v for fn, v in d.iteritems() if fn in allowed_key_list}

        def convert_keys_fn_to_sec(d):
            return {self.valid_filenames_to_sections[fn]: v for fn, v in d.iteritems()}

        try:
            auto_submasks = load_submasks(submasks_rootdir=submasks_rootdir)
            self.auto_submasks = convert_keys_fn_to_sec(filter_by_keys(auto_submasks, self.valid_filenames))
        except:
            self.auto_submasks = {}

        # # If user decisions exist
        # auto_submask_decisions = generate_submask_review_results(submasks_rootdir=submasks_rootdir, filenames=self.valid_filenames, which='user')
        # auto_submask_decisions = convert_keys_fn_to_sec(filter_by_keys(auto_submask_decisions, self.valid_filenames))
        # for sec, decisions in auto_submask_decisions.iteritems():
        #     if sec not in self.auto_submask_decisions or len(self.auto_submask_decisions[sec]) == 0:
        #         self.auto_submask_decisions[sec] = decisions

        # If no user decisions, load auto decisions
        auto_submask_decisions = generate_submask_review_results(submasks_rootdir=submasks_rootdir, filenames=self.valid_filenames, which='auto')
        auto_submask_decisions = convert_keys_fn_to_sec(filter_by_keys(auto_submask_decisions, self.valid_filenames))
        for sec, decisions in auto_submask_decisions.iteritems():
            if sec not in self.auto_submask_decisions or len(self.auto_submask_decisions[sec]) == 0:
                self.auto_submask_decisions[sec] = decisions
        self.auto_submasks_gscene.set_submasks_and_decisions(self.auto_submasks, self.auto_submask_decisions)
        self.auto_submasks_gscene.set_active_section(345, emit_changed_signal=False)

        ##########################
        ## User Submasks Gscene ##
        ##########################

        self.user_submasks_gscene = DrawableZoomableBrowsableGraphicsScene_ForMasking(id='userFinalMask', gview=self.ui.gview_final_masks_user)
        self.user_submasks_feeder = ImageDataFeeder(name='autoFinalMasks', stack=self.stack, \
                                    sections=self.valid_sections, use_data_manager=False, downscale=32,
                                    labeled_filenames={sec: os.path.join(RAW_DATA_DIR, self.stack, fn + ".png")
                                        for sec, fn in self.valid_sections_to_filenames.iteritems()})
        self.user_submasks_gscene.set_data_feeder(self.user_submasks_feeder)
        self.user_submasks_gscene.submask_decision_updated.connect(self.user_submask_decision_updated)

        # Load modified submasks and submask decisions.
        user_modified_submasks_rootdir = create_if_not_exists(os.path.join(THUMBNAIL_DATA_DIR, self.stack, self.stack + '_submasks_user_modified'))

        user_submasks = load_submasks(submasks_rootdir=user_modified_submasks_rootdir)
        self.user_submasks = convert_keys_fn_to_sec(filter_by_keys(user_submasks, self.valid_filenames))

        user_submask_decisions = generate_submask_review_results(submasks_rootdir=user_modified_submasks_rootdir, filenames=self.valid_filenames, which='user')
        user_submask_decisions = convert_keys_fn_to_sec(filter_by_keys(user_submask_decisions, self.valid_filenames))
        for sec, decisions in user_submask_decisions.iteritems():
            if sec not in self.user_submask_decisions or len(self.user_submask_decisions[sec]) == 0:
                self.user_submask_decisions[sec] = decisions

        selected_snake_lambda1, selected_dissim_thresholds, selected_channels = load_masking_parameters(submasks_rootdir=user_modified_submasks_rootdir)
        selected_snake_lambda1 = convert_keys_fn_to_sec(selected_snake_lambda1)
        selected_dissim_thresholds = convert_keys_fn_to_sec(selected_dissim_thresholds)
        selected_channels = convert_keys_fn_to_sec(selected_channels)

        for sec, v in selected_snake_lambda1.iteritems():
            self.selected_snake_lambda1[sec] = v
        for sec, th in selected_dissim_thresholds.iteritems():
            self.selected_dissim_thresholds[sec] = th
        for sec, ch in selected_channels.iteritems():
            self.selected_channels[sec] = ch

        self.user_submasks_gscene.set_submasks_and_decisions(self.user_submasks, self.user_submask_decisions)

        #########################################################

        self.ui.comboBox_channel.activated.connect(self.channel_changed)
        self.ui.comboBox_channel.addItems(['Red', 'Green', 'Blue'])

        #########################################################

        self.gscene_thresholded = ZoomableBrowsableGraphicsScene(id='thresholded', gview=self.ui.gview_thresholded)
        self.thresholded_image_feeder = ImageDataFeeder(name='thresholded', stack=self.stack, \
                                                        sections=self.valid_sections, use_data_manager=False,
                                                        downscale=32)
        self.gscene_thresholded.set_data_feeder(self.thresholded_image_feeder)

        #########################################################

        self.gscene_slic = ZoomableBrowsableGraphicsScene(id='slic', gview=self.ui.gview_slic)
        self.slic_image_feeder = ImageDataFeeder(name='slic', stack=self.stack, \
                                                sections=self.valid_sections, use_data_manager=False,
                                                downscale=32)
        self.gscene_slic.set_data_feeder(self.slic_image_feeder)

        #########################################################

        # self.gscene_dissimmap = ZoomableBrowsableGraphicsScene(id='dissimmap', gview=self.ui.gview_dissimmap)
        # self.dissim_image_feeder = ImageDataFeeder(name='dissimmap', stack=self.stack, \
        #                                         sections=self.valid_sections, use_data_manager=False,
        #                                         downscale=32)
        # self.gscene_dissimmap.set_data_feeder(self.dissim_image_feeder)

        #########################################################

        # self.init_user_submasks_gscene = ZoomableBrowsableGraphicsScene(id='submasks', gview=self.ui.gview_submasks)
        # self.submask_image_feeder = ImageDataFeeder(name='submasks', stack=self.stack, \
        #                                         sections=self.valid_sections, use_data_manager=False,
        #                                         downscale=32)
        # self.init_user_submasks_gscene.set_data_feeder(self.submask_image_feeder)

        self.init_user_submasks_gscene = DrawableZoomableBrowsableGraphicsScene_ForMasking(id='submasks', gview=self.ui.gview_submasks)
        self.init_user_submasks_image_feeder = ImageDataFeeder(name='submasks', stack=self.stack, \
                                                sections=self.valid_sections, use_data_manager=False,
                                                downscale=32)
        self.init_user_submasks_gscene.set_data_feeder(self.init_user_submasks_image_feeder)

        #########################################################

        self.gscene_merged_mask = ZoomableBrowsableGraphicsScene(id='mergedMask', gview=self.ui.gview_merged_mask)
        self.merged_masks_feeder = ImageDataFeeder(name='mergedMask', stack=self.stack, \
                                                sections=self.valid_sections, use_data_manager=False,
                                                downscale=32)
        self.gscene_merged_mask.set_data_feeder(self.merged_masks_feeder)

        #########################################################

        for sec in self.valid_sections:
            self.update_merged_mask(sec=sec)

        #########################################################

        self.dialog.showMaximized()

    def upload_masks(self):
        transfer_data_synced(fp_relative=os.path.join(self.stack, self.stack + '_masks'),
                            from_hostname='localhost', to_hostname='dm')

    def save_all(self):

        for sec in self.valid_sections:
            self.save(sec=sec)

        self.save_final_decisions()
        self.export_final_masks()
        sys.stderr.write('Saving all masks: Done.\n')

    def save_final_decisions(self):

        accept_which_fp = os.path.join(THUMBNAIL_DATA_DIR, self.stack, self.stack + '_submasks_finalDecisions.txt')
        with open(accept_which_fp, 'w') as f:
            for sec, accept_which in sorted(self.accept_which.items()):
                fn = self.valid_sections_to_filenames[sec]
                if accept_which == 0:
                    if sec not in self.auto_submask_decisions:
                        sys.stderr.write('No auto submask decisions for %s (%d)\n' % (fn, sec))
                        continue
                    decisions = self.auto_submask_decisions[sec]
                    if len(decisions) == 0:
                        sys.stderr.write('Auto submask decisions are empty for %s (%d)\n' % (fn, sec))
                elif accept_which == 1:
                    if sec not in self.user_submask_decisions:
                        sys.stderr.write('No user submask decisions for %s (%d)\n' % (fn, sec))
                        continue
                    decisions = self.user_submask_decisions[sec]
                    if len(decisions) == 0:
                        sys.stderr.write('User submask decisions are empty for %s (%d)\n' % (fn, sec))
                else:
                    raise

                f.write('%d %s %d %s\n' % (sec, fn, accept_which, ' '.join(map(lambda x: str(int(x)), decisions))))


    def save_submasks_and_decisions(self, submasks_dir, which, fn=None, sec=None):
        """
        If which is auto, save submasks, submask decisions and parameters to both modified and final folder.
        If which is user, save submasks and submask decisions to both modified and final folder.
        """

        if which == 'auto':
            submasks = self.auto_submasks
            submask_decisions = self.auto_submask_decisions
        elif which == 'user':
            submasks = self.user_submasks
            submask_decisions = self.user_submask_decisions
        else:
            raise

        if sec is None:
            sec = self.auto_submasks_gscene.active_section
        if sec not in submasks or sec not in submask_decisions:
            return
        fn = self.valid_sections_to_filenames[sec]

        submask_fn_dir = os.path.join(submasks_dir, fn)
        execute_command('rm -rf %(d)s; mkdir -p %(d)s' % {'d': submask_fn_dir})

        # Save submasks
        for submask_ind, m in enumerate(submasks[sec]):
            submask_fp = os.path.join(submask_fn_dir, fn + '_submask_%d.png' % submask_ind)
            imsave(submask_fp, np.uint8(m)*255)

        # Save submask decisions
        decisions_fp = os.path.join(submask_fn_dir, fn + '_submasksUserReview.txt')
        np.savetxt(decisions_fp, submask_decisions[sec], fmt='%d')

        # Save masking parameters
        if which == 'user':
            if sec in self.selected_dissim_thresholds or \
                sec in self.selected_channels or \
                sec in self.selected_snake_lambda1:

                parameters_fp = os.path.join(submask_fn_dir, fn + '_maskingParameters.txt')
                with open(parameters_fp, 'w') as f:
                    if sec in self.selected_snake_lambda1:
                        f.write('snake_lambda1 %d\n' % self.selected_snake_lambda1[sec])
                    if sec in self.selected_dissim_thresholds:
                        f.write('dissim_threshold %.2f\n' % self.selected_dissim_thresholds[sec])
                    if sec in self.selected_channels:
                        f.write('channel %d\n' % self.selected_channels[sec])
            else:
                sys.stderr.write('Parameters for %s(%d) is not saved (no modification made ?)\n' % (fn, sec))

    def save(self, sec=None, fn=None):

        if sec is None:
            sec = self.auto_submasks_gscene.active_section
        accept_which = ['auto', 'user'][self.accept_which[sec]]

        if accept_which == 'user':
            submasks_dir = create_if_not_exists(os.path.join(THUMBNAIL_DATA_DIR, self.stack, self.stack + '_submasks_user_modified'))
            self.save_submasks_and_decisions(submasks_dir=submasks_dir, sec=sec, which=accept_which)

    def set_accept_auto_to_true(self):
        sec = self.auto_submasks_gscene.active_section
        assert sec in self.auto_submasks
        # Clear later stage images.
        self.accepted_final_masks[sec] = self.auto_submasks[sec]
        self.accept_which[sec] = 0 # change to accept auto
        self.ui.button_toggle_accept_auto.setText(STR_USING_AUTO)

    def set_accept_auto_to_false(self):
        sec = self.auto_submasks_gscene.active_section
        assert sec in self.user_submasks
        self.accepted_final_masks[sec] = self.user_submasks[sec]
        self.accept_which[sec] = 1 # change to accept modified
        self.ui.button_toggle_accept_auto.setText(STR_USING_USER)

    def toggle_accept_auto(self):

        sec = self.auto_submasks_gscene.active_section

        if self.accept_which[sec] == 0: # currently accepting auto
            self.set_accept_auto_to_false()
        elif self.accept_which[sec] == 1: # currently accepting modified
            self.set_accept_auto_to_true()

        self.update_merged_mask()

    def export_final_masks(self):

        final_masks_dir = create_if_not_exists(os.path.join(THUMBNAIL_DATA_DIR, self.stack, self.stack + '_masks'))
        for sec, mask_viz in self.merged_mask_vizs.iteritems():
            fn = self.valid_sections_to_filenames[sec]
            imsave(os.path.join(final_masks_dir, fn + '_mask.png'), mask_viz)

    @pyqtSlot(int)
    def user_submask_decision_updated(self, submask_ind):
        self.update_merged_mask()
        self.update_mask_gui_window_title()

    @pyqtSlot(int)
    def auto_submask_decision_updated(self, submask_ind):
        self.update_merged_mask()
        self.update_mask_gui_window_title()

    @pyqtSlot()
    def update_merged_mask_clicked(self):
        self.update_merged_mask()

    def update_merged_mask(self, sec=None):
        """
        Update merged mask. Change the image shown in "Merged Mask" panel.
        """

        if sec is None:
            sec = self.auto_submasks_gscene.active_section
        fn = self.valid_sections_to_filenames[sec]
        accept_which = self.accept_which[sec]
        try:
            if accept_which == 0:
                if sec not in self.auto_submask_decisions or len(self.auto_submask_decisions[sec]) == 0:
                    # sys.stderr.write('Error: section %d, %s, accept auto but auto decisions is empty.\n' % (sec, fn))
                    raise Exception('Error: section %d, %s, accept auto but auto decisions is empty.' % (sec, fn))
                merged_mask = np.any([self.auto_submasks[sec][si] for si, dec in enumerate(self.auto_submask_decisions[sec]) if dec], axis=0)
            elif accept_which == 1:
                if sec not in self.user_submask_decisions or len(self.user_submask_decisions[sec]) == 0:
                    # sys.stderr.write('Error: section %d, %s, accept user but user decisions is empty.\n' % (sec, fn))
                    raise Exception('Error: section %d, %s, accept user but user decisions is empty.' % (sec, fn))
                merged_mask = np.any([self.user_submasks[sec][si] for si, dec in enumerate(self.user_submask_decisions[sec]) if dec], axis=0)
            else:
                raise Exception('accept_which is neither 0 or 1.')
            self.merged_masks[sec] = merged_mask
            self.merged_mask_vizs[sec] = img_as_ubyte(self.merged_masks[sec])
            self.merged_masks_feeder.set_image(sec=sec, numpy_image=self.merged_mask_vizs[sec])
            self.gscene_merged_mask.update_image(sec=sec)
        except Exception as e:
            sys.stderr.write('%s\n' % e)
            # raise e

    def update_slic(self):
        sec = self.auto_submasks_gscene.active_section

        t = time.time()
        self.slic_labelmaps[sec] = slic(self.contrast_stretched_images[sec].astype(np.float),
                                    sigma=SLIC_SIGMA, compactness=SLIC_COMPACTNESS,
                                    n_segments=SLIC_N_SEGMENTS, multichannel=False, max_iter=SLIC_MAXITER)
        sys.stderr.write('SLIC: %.2f seconds.\n' % (time.time() - t)) # 10 seconds, iter=100, nseg=1000;

        self.slic_boundary_images[sec] = img_as_ubyte(mark_boundaries(self.contrast_stretched_images[sec],
                                            label_img=self.slic_labelmaps[sec],
                                            background_label=-1, color=(1,0,0)))

        self.slic_image_feeder.set_image(sec=sec, numpy_image=self.slic_boundary_images[sec])
        self.gscene_slic.update_image(sec=sec)

        ####

        # self.ncut_labelmaps[sec] = normalized_cut_superpixels(self.contrast_stretched_images[sec], self.slic_labelmaps[sec])

        self.ncut_labelmaps[sec] = self.slic_labelmaps[sec]
        self.sp_dissim_maps[sec] = compute_sp_dissims_to_border(self.contrast_stretched_images[sec], self.ncut_labelmaps[sec])
        # self.sp_dissim_maps[sec] = compute_sp_dissims_to_border(self.thresholded_images[sec], self.ncut_labelmaps[sec])
        # self.border_dissim_images[sec] = generate_dissim_viz(self.sp_dissim_maps[sec], self.ncut_labelmaps[sec])
        # self.dissim_image_feeder.set_image(sec=sec, numpy_image=self.border_dissim_images[sec])
        # self.gscene_dissimmap.update_image(sec=sec)

        self.selected_dissim_thresholds[sec] = determine_dissim_threshold(self.sp_dissim_maps[sec], self.ncut_labelmaps[sec])
        self.ui.slider_dissimThresh.setValue(int(self.selected_dissim_thresholds[sec]/0.01))

        ######################################################

        self.update_init_submasks_image()

    def update_init_submasks_image(self):
        """
        Update the initial submasks for snake.
        """

        sec = self.auto_submasks_gscene.active_section

        self.selected_dissim_thresholds[sec] = self.ui.slider_dissimThresh.value() * 0.01
        # self.init_submasks[sec] = get_submasks(self.thresholded_images[sec])
        # self.init_submasks[sec] = get_submasks(ncut_labels=self.ncut_labelmaps[sec], sp_dissims=self.sp_dissim_maps[sec], dissim_thresh=self.selected_dissim_thresholds[sec])
        self.init_submasks[sec] = merge_overlapping_masks(get_submasks(ncut_labels=self.ncut_labelmaps[sec], sp_dissims=self.sp_dissim_maps[sec], dissim_thresh=self.selected_dissim_thresholds[sec]))
        self.init_submasks_vizs[sec] = generate_submasks_viz(self.contrast_stretched_images[sec], self.init_submasks[sec], color=(255,0,0))
        # self.init_submasks_vizs[sec] = generate_submasks_viz(self.thresholded_images[sec], self.init_submasks[sec], color=(255,0,0))
        self.init_user_submasks_image_feeder.set_image(sec=sec, numpy_image=self.init_submasks_vizs[sec])
        self.init_user_submasks_gscene.update_image(sec=sec)

        if sec in self.selected_snake_lambda1:
            self.ui.slider_snakeShrink.setValue(self.selected_snake_lambda1[sec])
        else:
            self.ui.slider_snakeShrink.setValue(MORPHSNAKE_LAMBDA1)

    def update_user_submasks_image(self):

        sec = self.auto_submasks_gscene.active_section
        self.user_submasks_gscene.remove_submask_and_decisions_for_one_section(sec=sec)
        self.selected_snake_lambda1[sec] = self.ui.slider_snakeShrink.value()
        # self.user_submasks[sec] = snake(img=self.original_images[sec], submasks=self.init_submasks[sec])
        submasks = snake(img=self.contrast_stretched_images[sec], submasks=self.init_submasks[sec],
                                        lambda1=self.selected_snake_lambda1[sec])
        if len(submasks) == 0:
            return
        else:
            self.user_submasks[sec] = submasks
        # self.user_submasks[sec] = snake(img=self.thresholded_images[sec], submasks=self.init_submasks[sec])
        # self.final_submasks_vizs[sec] = generate_submasks_viz(self.original_images[sec], self.user_submasks[sec], color=(255,0,0))
        self.user_submasks_gscene.update_image(sec=sec)
        self.user_submask_decisions[sec] = auto_judge_submasks(self.user_submasks[sec])
        self.user_submasks_gscene.add_submask_and_decision_for_one_section(submasks=self.user_submasks[sec],
        submask_decisions=self.user_submask_decisions[sec], sec=sec)

        self.set_accept_auto_to_false()
        self.update_merged_mask()

    def change_channel(self, channel):
        print 'Changed to', channel
        sec = self.auto_submasks_gscene.active_section
        self.contrast_stretched_images[sec] = contrast_stretch_image(self.original_images[sec][..., channel])
        self.update_thresholded_image()

    def channel_changed(self, index):
        self.selected_channels[self.auto_submasks_gscene.active_section] = index

        channel_text = str(self.sender().currentText())
        if channel_text == 'Red':
            self.change_channel(0)
        elif channel_text == 'Green':
            self.change_channel(1)
        elif channel_text == 'Blue':
            self.change_channel(2)

    def dissim_threshold_changed(self, value):
        self.ui.label_dissimThresh.setText(str(value * 0.01))

    def snake_shrinkParam_changed(self, value):
        self.ui.label_snakeShrink.setText(str(value))

    def update_thresholded_image(self):
        print "update_thresholded_image"
        sec = self.auto_submasks_gscene.active_section
        # thresholded_image = (self.contrast_stretched_images[sec] < self.selected_thresholds[sec]).astype(np.uint8)*255
        # thresholded_image = img_as_ubyte(remove_small_holes(thresholded_image, min_size=50, connectivity=1))
        # self.thresholded_images[sec] = thresholded_image
        self.thresholded_images[sec] = self.contrast_stretched_images[sec]
        self.thresholded_image_feeder.set_image(sec=sec, qimage=numpy_to_qimage(self.thresholded_images[sec]))

        self.gscene_thresholded.update_image(sec=sec)

    def auto_submasks_gscene_section_changed(self):
        """
        What happens when the image in "Automatic Masks" panel is changed.
        """

        self.update_mask_gui_window_title()

        sec = self.auto_submasks_gscene.active_section

        if sec not in self.contrast_stretched_images:
            if sec not in self.original_images:
                img = imread(DataManager.get_image_filepath(stack=self.stack, section=sec, resol='thumbnail', version='original_png'))
                border = np.median(np.concatenate([img[:10, :].flatten(), img[-10:, :].flatten(), img[:, :10].flatten(), img[:, -10:].flatten()]))
                if border < 123:
                    # dark background, fluorescent
                    img = img.max() - img # invert, make tissue dark on bright background
                self.original_images[sec] = img

        if self.accept_which[sec] == 1:
            self.ui.button_toggle_accept_auto.setText(STR_USING_USER)
        elif self.accept_which[sec] == 0:
            self.ui.button_toggle_accept_auto.setText(STR_USING_AUTO)

        # Set parameters if those for the current section have been modified before.

        if sec not in self.selected_channels:
            self.selected_channels[sec] = 0

        self.ui.comboBox_channel.setCurrentIndex(self.selected_channels[sec])
        self.change_channel(self.selected_channels[sec])

        try:
            self.gscene_thresholded.set_active_section(sec)
        except: # The first time this will complain "Image not loaded" yet. But will not once update_thresholded_image() loads the image.
            pass

        self.update_thresholded_image()

        try:
            self.gscene_slic.set_active_section(sec)
        except:
            pass

        try:
            self.init_user_submasks_gscene.set_active_section(sec)
        except:
            pass

        if sec in self.selected_dissim_thresholds:
            self.ui.slider_dissimThresh.setValue(int(self.selected_dissim_thresholds[sec]/0.01))
        else:
            self.ui.slider_dissimThresh.setValue(0)

        if sec in self.selected_snake_lambda1:
            self.ui.slider_snakeShrink.setValue(self.selected_snake_lambda1[sec])
        else:
            self.ui.slider_snakeShrink.setValue(MORPHSNAKE_LAMBDA1)

        try:
            self.user_submasks_gscene.set_active_section(sec)
        except:
            pass

        try:
            self.gscene_merged_mask.set_active_section(sec)
        except:
            pass

    def update_mask_gui_window_title(self):
        curr_sec = self.auto_submasks_gscene.active_section
        curr_fn = self.valid_sections_to_filenames[curr_sec]
        title = '%s (%d) - Active: %s - Alg:%s User:%s' % (curr_fn, curr_sec, ['Alg', 'User'][self.accept_which[curr_sec]], self.auto_submask_decisions[curr_sec], self.user_submask_decisions[curr_sec])
        self.dialog.setWindowTitle(title)
        print title

    def generate_masks(self):
        web_services_request('generate_masks', stack=self.stack, filenames=self.valid_filenames, tb_fmt='png')
        transfer_data_synced(fp_relative=os.path.join(self.stack, self.stack + '_masks'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Mask Editing GUI')
    parser.add_argument("stack", type=str, help="stack name")
    args = parser.parse_args()
    stack = args.stack

    app = QApplication(argv)

    m = MaskEditingGUI(stack=stack)
    # m.showMaximized()
    m.show()
    exit(app.exec_())
