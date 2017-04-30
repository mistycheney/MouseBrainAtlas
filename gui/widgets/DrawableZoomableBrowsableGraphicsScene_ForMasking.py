import sys, os
from collections import defaultdict

from PyQt4.QtCore import *
from PyQt4.QtGui import *

sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from data_manager import DataManager
from metadata import *
from gui_utilities import *
from registration_utilities import find_contour_points
from annotation_utilities import contours_to_mask

from DrawableZoomableBrowsableGraphicsScene import DrawableZoomableBrowsableGraphicsScene

class DrawableZoomableBrowsableGraphicsScene_ForMasking(DrawableZoomableBrowsableGraphicsScene):
    """
    Extends base class by:
    - defining specific signal handlers
    - adding a dict called submask_decisions
    """

    submask_decision_updated = pyqtSignal(int)

    def __init__(self, id, gview=None, parent=None):
        super(DrawableZoomableBrowsableGraphicsScene_ForMasking, self).__init__(id=id, gview=gview, parent=parent)

        self.set_default_line_color('g')
        self.set_default_line_width(1)
        self.set_default_vertex_color('b')
        self.set_default_vertex_radius(2)

        self.polygon_completed.connect(self.submask_added)
        self.polygon_pressed.connect(self.submask_clicked)
        self.polygon_deleted.connect(self.submask_deleted)

        # self._submasks = {}
        self._submask_decisions = defaultdict(dict)

    def set_submasks_and_decisions(self, submasks, submask_decisions):
        for sec in submasks.iterkeys():
            self.set_submasks_and_decisions_one_section(sec, submasks[sec], submask_decisions[sec])

    def set_submasks_and_decisions_one_section(self, sec, submasks, submask_decisions):
        self.delete_all_polygons_one_section(section=sec)
        self.add_submasks_and_decisions_one_section(sec=sec, submasks=submasks, submask_decisions=submask_decisions)

    # def set_submask(self, sec, submask_ind, submask):
    #     self._submasks[sec][submask_ind] = submask

    def update_color_from_submask_decision(self, sec, submask_ind):
        index, _ = self.get_requested_index_and_section(sec=sec)
        if self._submask_decisions[sec][submask_ind]:
            pen = QPen(Qt.green)
        else:
            pen = QPen(Qt.red)
        self.drawings[index][submask_ind].setPen(pen)

    def set_submask_decision(self, sec, submask_ind, decision):
        self._submask_decisions[sec][submask_ind] = decision
        self.update_color_from_submask_decision(sec, submask_ind)
        self.submask_decision_updated.emit(submask_ind)

    # def remove_submask_and_decisions_for_one_section(self, sec):
    #     if sec in self.submasks:
    #         del self.submasks[sec]
    #     if sec in self.submask_decisions:
    #         del self.submask_decisions[sec]
    #     self.delete_all_polygons_one_section(section=sec)

    # def add_submasks_and_decisions(self, submasks, submask_decisions):
    #     for sec in submasks.iterkeys():
    #         self.add_submasks_and_decisions_one_section(sec, submasks[sec], submask_decisions[sec])
    #
    def add_submasks_and_decisions_one_section(self, sec, submasks, submask_decisions):

        # if sec not in submask_decisions:
        #     raise Exception("Cannot update image for section %d because no submasks exist." % sec)
            # sys.stderr.write("Cannot update image for section %d because no submasks exist.\n" % sec)
            # return

        for submask_ind, decision in submask_decisions.iteritems():
            m = submasks[submask_ind]
            cnts = find_contour_points(m, sample_every=1)[m.max()]
            if len(cnts) == 0:
                raise Exception('ERROR: section %d %d, submask %d - no contour' % (sec, submask_ind, len(cnts)))
            elif len(cnts) > 1:
                sys.stderr.write('WARNING: section %d, submask %d - %d contours\n' % (sec, submask_ind, len(cnts)))
                cnt = sorted(cnts, key=lambda c: len(c), reverse=True)[0]
            else:
                cnt = cnts[0]

            self._submask_decisions[sec][submask_ind] = decision

            if decision:
                color = 'g'
            else:
                color = 'r'

            self.add_polygon(path=vertices_to_path(cnt), section=sec, linewidth=2, color=color)

    # def add_submask_and_decision_for_one_section(self, submasks, submask_decisions, sec):
    #     self.submasks[sec] = submasks
    #     self.submask_decisions[sec] = submask_decisions
    #
    #     for submask_ind, decision in enumerate(submask_decisions):
    #
    #         assert submask_ind < len(submasks), 'Error: decisions loaded for section %d, submask %d (start from 0), but not submask image' % (sec, submask_ind)
    #         mask = submasks[submask_ind]
    #
    #         # Compute contour of mask
    #         cnts = find_contour_points(mask, sample_every=1)[mask.max()]
    #         if len(cnts) == 0:
    #             raise Exception('ERROR: section %d %d, submask %d - no contour' % (sec, submask_ind, len(cnts)))
    #         elif len(cnts) > 1:
    #             sys.stderr.write('WARNING: section %d, submask %d - %d contours\n' % (sec, submask_ind, len(cnts)))
    #             cnt = sorted(cnts, key=lambda c: len(c), reverse=True)[0]
    #         else:
    #             cnt = cnts[0]
    #
    #         if decision:
    #             color = 'g'
    #         else:
    #             color = 'r'
    #
    #         self.add_polygon(path=vertices_to_path(cnt), section=sec, linewidth=2, color=color)

    # def set_submasks_and_decisions(self, submasks_allFiles, submask_decisions_allFiles):
    #     """
    #     These two inputs will be modified by actions on the gscene (e.g. clicking on polygons).
    #     """
    #
    #     self.submasks = submasks_allFiles
    #     self.submask_decisions = submask_decisions_allFiles

        # for sec, submasks in submasks_allFiles.iteritems():
        #     self.add_submask_and_decision_for_one_section(submasks, submask_decisions_allFiles[sec], sec=sec)

    def submask_added(self, polygon):
        pass
        # if self.active_section not in self.submask_decisions:
        #     self.submask_decisions[self.active_section] = []
        # n = len(self.submask_decisions[self.active_section])
        # existing_submask_indices = range(n)
        # if len(existing_submask_indices) == 0:
        #     first_available_submask_ind = 0
        # else:
        #     available_submask_indices = [i for i in range(np.max(existing_submask_indices) + 2) if i not in existing_submask_indices]
        #     first_available_submask_ind = available_submask_indices[0]
        # new_submask_ind = first_available_submask_ind
        #
        # # Update submask decisions
        # if new_submask_ind <= n-1:
        #     self.submask_decisions[self.active_section][new_submask_ind] = True
        # else:
        #     self.submask_decisions[self.active_section].append(True)
        #
        # # Update submask list
        # new_submask_contours = vertices_from_polygon(polygon)
        #
        # bbox = self.pixmapItem.boundingRect()
        # image_shape = (int(bbox.height()), int(bbox.width()))
        # new_submask = contours_to_mask([new_submask_contours], image_shape)
        # if self.active_section not in self.submasks:
        #     self.submasks[self.active_section] = []
        # if new_submask_ind <= n-1:
        #     self.submasks[self.active_section][new_submask_ind] = new_submask
        # else:
        #     self.submasks[self.active_section].append(new_submask)
        #
        # sys.stderr.write('Submask %d added.\n' % new_submask_ind)

    def submask_deleted(self, polygon, index, polygon_index):
        pass
        # submask_ind = polygon_index
        # del self.submask_decisions[self.active_section][submask_ind]
        # del self.submasks[self.active_section][submask_ind]
        # sys.stderr.write('Submask %d removed.\n' % submask_ind)
        # self.update_mask_gui_window_title()

    def submask_clicked(self, polygon):
        submask_ind = self.drawings[self.active_i].index(polygon)
        sys.stderr.write('Submask %d clicked.\n' % submask_ind)

        curr_decision = self._submask_decisions[self.active_section][submask_ind]
        self.set_submask_decision(submask_ind=submask_ind, decision=not curr_decision, sec=self.active_section)

    # def update_submask_decision(self, submask_ind, decision, sec):

        # if submask_ind in self.submask_decisions[sec]:
        #     curr_decision = self.submask_decisions[sec][submask_ind]
        #     if curr_decision == decision:
        #         return
        #
        # self.submask_decisions[sec][submask_ind] = decision
        #
        # if decision:
        #     pen = QPen(Qt.green)
        # else:
        #     pen = QPen(Qt.red)
        # pen.setWidth(self.default_line_width)
        #
        # curr_i, _ = self.get_requested_index_and_section(sec=sec)
        # self.drawings[curr_i][submask_ind].setPen(pen)

        # self.submask_decision_updated.emit(submask_ind)

    def accept_all_submasks(self):
        # for submask_ind in self.submasks[self.active_section].iterkeys():
        for submask_ind in range(len(self.submasks[self.active_section])):
            self.update_submask_decision(submask_ind=submask_ind, decision=True, sec=self.active_section)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Space:
                self.accept_all_submasks()
                return True

        return super(DrawableZoomableBrowsableGraphicsScene_ForMasking, self).eventFilter(obj, event)
