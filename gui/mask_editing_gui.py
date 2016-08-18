#! /usr/bin/env python

import sip
sip.setapi('QVariant', 2) # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string

from matplotlib.backends import qt4_compat
use_pyside = qt4_compat.QT_API == qt4_compat.QT_API_PYSIDE
if use_pyside:
    #print 'Using PySide'
    from PySide.QtCore import *
    from PySide.QtGui import *
else:
    #print 'Using PyQt4'
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *

from ui_MaskEditingGui_v1 import Ui_MaskEditingGui

import cPickle as pickle

import sys, os
sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *

from DataFeeder import ImageDataFeeder, VolumeResectionDataFeeder
from drawable_gscene import *

from gui_utilities import *

# Use the third method in http://pyqt.sourceforge.net/Docs/PyQt4/designer.html
class MaskEditingGUI(QMainWindow, Ui_MaskEditingGui):
    def __init__(self, parent=None, stack=None):
        """
        Initialization of RectificationTool.
        """
        # self.app = QApplication(sys.argv)
        QMainWindow.__init__(self, parent)

        self.setupUi(self)

        self.stack = stack

        # self.volume_cache = {32: bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_down%(downsample)dVolume.bp' % {'stack':self.stack, 'downsample':32}),
        #                     8: bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_down%(downsample)dVolume.bp' % {'stack':self.stack, 'downsample':8})}

        # self.volume_cache = {32: bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_down%(downsample)dVolume.bp' % {'stack':self.stack, 'downsample':32})}
                            # 8: bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_down%(downsample)dVolume.bp' % {'stack':self.stack, 'downsample':8})}

        self.cerebellum_masks = {}

        # self.volume = self.volume_cache[self.downsample_factor]
        # self.y_dim, self.x_dim, self.z_dim = self.volume.shape

        self.section_gscene = DrawableGraphicsScene(id='sagittal', gui=self, gview=self.section_gview)
        self.section_gview.setScene(self.section_gscene)

        first_sec, last_sec = section_range_lookup[self.stack]
        self.sections = range(first_sec, last_sec+1)
        image_feeder = ImageDataFeeder('image feeder', stack=self.stack, sections=self.sections)
        image_feeder.set_downsample_factor(32)
        image_feeder.set_orientation('sagittal')
        image_feeder.load_images()
        self.section_gscene.set_data_feeder(image_feeder)
        self.section_gscene.set_active_section((first_sec + last_sec)/2)

        self.section_gscene.set_vertex_radius(5)

        from functools import partial
        self.section_gscene.set_conversion_func_section_to_z(partial(DataManager.convert_section_to_z, stack=self.stack))
        self.section_gscene.set_conversion_func_z_to_section(partial(DataManager.convert_z_to_section, stack=self.stack))

        self.installEventFilter(self)

        self.button_save.clicked.connect(self.save)
        self.button_load.clicked.connect(self.load)

    def eventFilter(self, obj, event):

        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_1:
                self.section_gscene.show_previous()
            elif key == Qt.Key_2:
                self.section_gscene.show_next()

        return False


    def load(self):
        # self.gscenes['sagittal'].load_drawings(username='Lauren', timestamp='latest', annotation_rootdir=annotation_midbrainIncluded_v2_rootdir)
        self.section_gscene.load_drawings(username='yuncong', timestamp='latest', annotation_rootdir=cerebellum_masks_rootdir)

    @pyqtSlot()
    def save(self):

        # if not hasattr(self, 'username') or self.username is None:
        #     username, okay = QInputDialog.getText(self, "Username", "Please enter your username:", QLineEdit.Normal, 'anon')
        #     if not okay: return
        #     self.username = str(username)
        #     self.lineEdit_username.setText(self.username)

        self.username = 'yuncong'

        # labelings_dir = create_if_not_exists('/home/yuncong/CSHL_labelings_new/%(stack)s/' % dict(stack=self.stack))
        labelings_dir = create_if_not_exists(os.path.join(cerebellum_masks_rootdir, stack))

        timestamp = datetime.datetime.now().strftime("%m%d%Y%H%M%S")

        self.section_gscene.save_drawings(fn_template=os.path.join(labelings_dir, '%(stack)s_%(orientation)s_downsample%(downsample)d_%(username)s_%(timestamp)s.pkl'), timestamp=timestamp, username=self.username)

        self.statusBar().showMessage('Labelings saved to %s.' % labelings_dir)


if __name__ == "__main__":

    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Mask Editing GUI.')

    parser.add_argument("stack_name", type=str, help="stack name")
    args = parser.parse_args()

    from sys import argv, exit
    app = QApplication(argv)

    m = MaskEditingGUI(stack=args.stack_name)

    # m.show()
    m.showMaximized()
    # m.raise_()
    exit(app.exec_())
