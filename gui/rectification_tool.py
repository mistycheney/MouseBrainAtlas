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

from ui_RectificationTool import Ui_RectificationGUI

import cPickle as pickle

import sys, os
sys.path.append(os.environ['REPO_DIR'] + '/utilities')
from utilities2015 import *
from metadata import *
from data_manager import *

from gui_utilities import *

CROSSLINE_PEN_WIDTH = 2

crossline_red_pen = QPen(Qt.red)
crossline_red_pen.setWidth(CROSSLINE_PEN_WIDTH)

# Use the third method in http://pyqt.sourceforge.net/Docs/PyQt4/designer.html
class RectificationTool(QMainWindow, Ui_RectificationGUI):
    def __init__(self, parent=None, stack=None):
        """
        Initialization of RectificationTool.
        """
        # self.app = QApplication(sys.argv)
        QMainWindow.__init__(self, parent)

        self.setupUi(self)

        self.stack = stack

        self.volume_cache = {32: bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_down%(downsample)dVolume.bp' % {'stack':self.stack, 'downsample':32}),
                            8: bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_down%(downsample)dVolume.bp' % {'stack':self.stack, 'downsample':8})}

        self.downsample_factor = 32

        self.volume = self.volume_cache[self.downsample_factor]
        self.y_dim, self.x_dim, self.z_dim = self.volume.shape

        self.coronal_gscene = QGraphicsScene()
        self.coronal_gview.setScene(self.coronal_gscene)

        self.horizontal_gscene = QGraphicsScene()
        self.horizontal_gview.setScene(self.horizontal_gscene)

        self.sagittal_gscene = QGraphicsScene()
        self.sagittal_gview.setScene(self.sagittal_gscene)

        self.coronal_gscene.installEventFilter(self)
        self.horizontal_gscene.installEventFilter(self)
        self.sagittal_gscene.installEventFilter(self)
        self.coronal_gview.viewport().installEventFilter(self)
        self.horizontal_gview.viewport().installEventFilter(self)
        self.sagittal_gview.viewport().installEventFilter(self)

        # self.sagittal_gview.viewport().installEventFilter(self)

        self.sagittal_pixmapItem = QGraphicsPixmapItem()
        self.sagittal_gscene.addItem(self.sagittal_pixmapItem)

        self.coronal_pixmapItem = QGraphicsPixmapItem()
        self.coronal_gscene.addItem(self.coronal_pixmapItem)

        self.horizontal_pixmapItem = QGraphicsPixmapItem()
        self.horizontal_gscene.addItem(self.horizontal_pixmapItem)

        self.coronal_hline = QGraphicsLineItem()
        self.coronal_hline.setPen(crossline_red_pen)
        # self.coronal_hline.setZValue(0)
        self.coronal_vline = QGraphicsLineItem()
        self.coronal_vline.setPen(crossline_red_pen)
        # self.coronal_vline.setZValue(0)
        self.horizontal_hline = QGraphicsLineItem()
        self.horizontal_hline.setPen(crossline_red_pen)
        # self.horizontal_hline.setZValue(0)
        self.horizontal_vline = QGraphicsLineItem()
        self.horizontal_vline.setPen(crossline_red_pen)
        # self.horizontal_vline.setZValue(0)
        self.sagittal_hline = QGraphicsLineItem()
        self.sagittal_hline.setPen(crossline_red_pen)
        # self.horizontal_hline.setZValue(0)
        self.sagittal_vline = QGraphicsLineItem()
        self.sagittal_vline.setPen(crossline_red_pen)
        # self.horizontal_vline.setZValue(0)

        self.coronal_gscene.addItem(self.coronal_hline)
        self.coronal_gscene.addItem(self.coronal_vline)
        self.horizontal_gscene.addItem(self.horizontal_hline)
        self.horizontal_gscene.addItem(self.horizontal_vline)
        self.sagittal_gscene.addItem(self.sagittal_hline)
        self.sagittal_gscene.addItem(self.sagittal_vline)

        # self.x_dim = 99
        # self.y_dim = 99
        # self.z_dim = 99

        # self.sagittal_pixmap = QPixmap()
        # self.sagittal_pixmapItem = self.sagittal_gscene.addPixmap(self.sagittal_pixmap)
        # self.sagittal_pixmapItem.setZValue(-10)

        # self.coronal_pixmap = QPixmap()
        # self.coronal_pixmapItem = self.coronal_gscene.addPixmap(self.coronal_pixmap)
        # # self.coronal_pixmapItem.setZValue(-10)
        #
        # self.horizontal_pixmap = QPixmap()
        # self.horizontal_pixmapItem = self.horizontal_gscene.addPixmap(self.horizontal_pixmap)
        # self.horizontal_pixmapItem.setZValue(-10)


        # self.sagittal_hline = self.coronal_gscene.addLine(0, self.y, self.coronal_gscene.width(), self.y)
        # self.sagittal_vline = self.coronal_gscene.addLine(0, self.y, self.coronal_gscene.width(), self.y)

        # self.button_sameX.clicked.connect(self.sameX_clicked)
        # self.button_sameY.clicked.connect(self.sameY_clicked)
        # self.button_sameZ.clicked.connect(self.sameZ_clicked)

        self.button_symmetric.clicked.connect(self.symmetric_clicked)
        self.button_midline.clicked.connect(self.midline_clicked)
        self.button_cancel.clicked.connect(self.cancel_clicked)

        self.slider_hxy.valueChanged.connect(self.slider_hxy_changed)
        self.slider_hxz.valueChanged.connect(self.slider_hxz_changed)
        self.slider_hyx.valueChanged.connect(self.slider_hyx_changed)
        self.slider_hyz.valueChanged.connect(self.slider_hyz_changed)
        self.slider_hzx.valueChanged.connect(self.slider_hzx_changed)
        self.slider_hzy.valueChanged.connect(self.slider_hzy_changed)

        self.slider_downsample.valueChanged.connect(self.downsample_factor_changed)

        self.button_done.clicked.connect(self.done_clicked)

        self.first_point_done = False
        self.mode = None
        self.recorded_points = {'symmetric': [], 'midline': []}

        self.update_cross(0,0,0)

        self.hxy = 0
        self.hxz = 0
        self.hyx = 0
        self.hyz = 0
        self.hzx = 0
        self.hzy = 0

    def update_crosslines(self):
        print self.cross_x, self.cross_y, self.cross_z
        self.coronal_hline.setLine(0, self.cross_y, self.z_dim-1, self.cross_y)
        self.coronal_vline.setLine(self.z_dim-1-self.cross_z, 0, self.z_dim-1-self.cross_z, self.y_dim-1)
        self.horizontal_hline.setLine(0, self.z_dim-1-self.cross_z, self.x_dim-1, self.z_dim-1-self.cross_z)
        self.horizontal_vline.setLine(self.cross_x, 0, self.cross_x, self.z_dim-1)
        self.sagittal_hline.setLine(0, self.cross_y, self.x_dim-1, self.cross_y)
        self.sagittal_vline.setLine(self.cross_x, 0, self.cross_x, self.y_dim-1)

    def done_clicked(self):
        resectioned_dir = create_if_not_exists('/home/yuncong/CSHL_volumes_resection/%(stack)s/' % {'stack':stack})
        pickle.dump(self.recorded_points, open(resectioned_dir + '/recorded_points.pkl' % {'stack':stack}, 'w'))

    # def sameX_clicked(self):
    #     self.mode = 'sameX'
    #
    # def sameY_clicked(self):
    #     self.mode = 'sameY'

    # def sameZ_clicked(self):
    #     self.mode = 'sameZ'

    def symmetric_clicked(self):
        self.mode = 'symmetric'

    def midline_clicked(self):
        self.mode = 'midline'

    def cancel_clicked(self):
        self.mode = None


    def update_gscenes(self, which):

        gray_color_table = [qRgb(i, i, i) for i in range(256)]

        if which == 'y':

            # color http://stackoverflow.com/questions/9794019/convert-numpy-array-to-pyside-qpixmap color comes out wrong
            # gray https://gist.github.com/smex/5287589 works fine; Must specify bytesPerLine, see http://doc.qt.io/qt-4.8/qimage.html#QImage-6

            # self.horizontal_data = self.volume[self.cross_y, :, :].flatten()
            # self.horizontal_image = QImage(self.horizontal_data, self.z_dim, self.x_dim, self.z_dim, QImage.Format_Indexed8)
            self.horizontal_data = self.volume[self.cross_y, :, ::-1].T.flatten()
            self.horizontal_image = QImage(self.horizontal_data, self.x_dim, self.z_dim, self.x_dim, QImage.Format_Indexed8)
            self.horizontal_image.setColorTable(gray_color_table)

            # a = gray2rgb(self.volume[self.cross_y, :, :])
            # self.horizontal_data = (255 << 24 | a[:,:,0] << 16 | a[:,:,1] << 8 | a[:,:,2]).flatten().copy() # pack RGB values
            # self.horizontal_image = QImage(self.horizontal_data, self.z_dim, self.x_dim, QImage.Format_RGB32)

            self.horizontal_pixmap = QPixmap.fromImage(self.horizontal_image)

            # horizontal_fn = '/home/yuncong/CSHL_volumes_resection/horizontal/%(stack)s_thumbnailVolume_horizontal_%(y)03d.tif' % {'stack': self.stack, 'y': self.cross_y}
            # self.horizontal_pixmap = QPixmap(horizontal_fn)
            self.horizontal_pixmapItem.setPixmap(self.horizontal_pixmap)

            # self.x_dim = self.horizontal_pixmap.height()
            # self.z_dim = self.horizontal_pixmap.width()

        elif which == 'x':
            # coronal_fn = '/home/yuncong/CSHL_volumes_resection/coronal/%(stack)s_thumbnailVolume_coronal_%(x)03d.tif' % {'stack': self.stack, 'x': self.cross_x}
            # self.coronal_pixmap = QPixmap(coronal_fn)

            # a = gray2rgb(self.volume[:, self.cross_x, :])
            # self.coronal_data = (255 << 24 | a[:,:,0] << 16 | a[:,:,1] << 8 | a[:,:,2]).flatten().copy() # pack RGB values
            # self.coronal_image = QImage(self.coronal_data, self.z_dim, self.y_dim, QImage.Format_RGB32)

            self.coronal_data = self.volume[:, self.cross_x, ::-1].flatten()
            self.coronal_image = QImage(self.coronal_data, self.z_dim, self.y_dim, self.z_dim, QImage.Format_Indexed8)
            self.coronal_image.setColorTable(gray_color_table)

            self.coronal_pixmap = QPixmap.fromImage(self.coronal_image)
            self.coronal_pixmapItem.setPixmap(self.coronal_pixmap)

            # self.z_dim = self.coronal_pixmap.width()
            # self.y_dim = self.coronal_pixmap.height()

        elif which == 'z':
            # sagittal_fn = '/home/yuncong/CSHL_volumes_resection/sagittal/%(stack)s_thumbnailVolume_sagittal_%(z)03d.tif' % {'stack': self.stack, 'z': self.cross_z}
            # self.sagittal_pixmap = QPixmap(sagittal_fn)

            # a = gray2rgb(self.volume[:, :, self.cross_z])
            # self.sagittal_data = (255 << 24 | a[:,:,0] << 16 | a[:,:,1] << 8 | a[:,:,2]).flatten().copy() # pack RGB values
            # self.sagittal_image = QImage(self.sagittal_data, self.x_dim, self.y_dim, QImage.Format_RGB32)

            self.sagittal_data = self.volume[:, :, self.cross_z].flatten()

            self.sagittal_image = QImage(self.sagittal_data, self.x_dim, self.y_dim, self.x_dim, QImage.Format_Indexed8)
            self.sagittal_image.setColorTable(gray_color_table)

            self.sagittal_pixmap = QPixmap.fromImage(self.sagittal_image)
            self.sagittal_pixmapItem.setPixmap(self.sagittal_pixmap)

            # self.x_dim = self.sagittal_pixmap.width()
            # self.y_dim = self.sagittal_pixmap.height()


    def update_cross(self, x=None, y=None, z=None):

        if y is not None:
            self.cross_y = max(0, min(self.y_dim-1, y))
            self.update_gscenes('y')
        if x is not None:
            self.cross_x = max(0, min(self.x_dim-1, x))
            self.update_gscenes('x')
        if z is not None:
            self.cross_z = max(0, min(self.z_dim-1, z))
            self.update_gscenes('z')

        self.update_crosslines()

    def translate_gsceneCoord_to_3d(self, which, gscene_x, gscene_y):
        if which == 'coronal':
            x = self.cross_x
            y = gscene_y
            z = self.z_dim - 1 - gscene_x
        elif which == 'horizontal':
            x = gscene_x
            y = self.cross_y
            z = self.z_dim - 1 - gscene_y
        elif which == 'sagittal':
            x = gscene_x
            y = gscene_y
            z = self.cross_z

        return (x,y,z)

    def slider_hxy_changed(self, val):
        self.hxy = val / 10.
        self.update_transform()

    def slider_hxz_changed(self, val):
        self.hxz = val / 10.
        self.update_transform()

    def slider_hyx_changed(self, val):
        self.hyx = val / 10.
        self.update_transform()

    def slider_hyz_changed(self, val):
        self.hyz = val / 10.
        self.update_transform()

    def slider_hzx_changed(self, val):
        self.hzx = val / 10.
        self.update_transform()

    def slider_hzy_changed(self, val):
        self.hzy = val / 10.
        self.update_transform()


    def downsample_factor_changed(self, val):

        self.cross_x_lossless = self.cross_x * self.downsample_factor
        self.cross_y_lossless = self.cross_y * self.downsample_factor
        self.cross_z_lossless = self.cross_z * self.downsample_factor

        print val

        if val == 0:
            self.downsample_factor = 32
        elif val == 1:
            self.downsample_factor = 8
        elif val == 2:
            self.downsample_factor = 4
            if 4 not in self.volume_cache:
                self.volume_cache[4] = bp.unpack_ndarray_file(volume_dir + '/%(stack)s/%(stack)s_down%(downsample)dVolume.bp' % {'stack':self.stack, 'downsample':4})
                self.statusBar().showMessage('Volume loaded.')

        self.volume = self.volume_cache[self.downsample_factor]
        self.y_dim, self.x_dim, self.z_dim = self.volume.shape

        self.update_cross(self.cross_x_lossless / self.downsample_factor, self.cross_y_lossless / self.downsample_factor, self.cross_z_lossless / self.downsample_factor)
        self.update_gscenes('x')
        self.update_gscenes('y')
        self.update_gscenes('z')

    def update_transform(self):
        T = np.array([[1, self.hxy, self.hxz], [self.hyx, 1, self.hyz], [self.hzx, self.hzy, 1]])


    def eventFilter(self, obj, event):

        event_type = event.type()

        if event_type == QEvent.GraphicsSceneWheel or event_type == QEvent.Wheel:

            if event.delta() < 0: # negative means towards user
                if obj == self.sagittal_gscene or obj == self.sagittal_gview.viewport():
                    self.update_cross(z=self.cross_z+1)
                elif obj == self.coronal_gscene or obj == self.coronal_gview.viewport():
                    self.update_cross(x=self.cross_x+1)
                elif obj == self.horizontal_gscene or obj == self.horizontal_gview.viewport():
                    self.update_cross(y=self.cross_y+1)
            else: # scroll away from user
                if obj == self.sagittal_gscene or obj == self.sagittal_gview.viewport():
                    self.update_cross(z=self.cross_z-1)
                elif obj == self.coronal_gscene or obj == self.coronal_gview.viewport():
                    self.update_cross(x=self.cross_x-1)
                elif obj == self.horizontal_gscene or obj == self.horizontal_gview.viewport():
                    self.update_cross(y=self.cross_y-1)

            return True

        if event_type == QEvent.GraphicsSceneMousePress:

            gscene_x = event.scenePos().x()
            gscene_y = event.scenePos().y()

            if obj == self.sagittal_gscene:
                self.active_view = 'sagittal'
            elif obj == self.coronal_gscene:
                self.active_view = 'coronal'
            elif obj == self.horizontal_gscene:
                self.active_view = 'horizontal'

            if self.mode is None:

                if obj == self.sagittal_gscene:
                    x, y, z = self.translate_gsceneCoord_to_3d('sagittal', gscene_x, gscene_y)
                elif obj == self.coronal_gscene:
                    x, y, z = self.translate_gsceneCoord_to_3d('coronal', gscene_x, gscene_y)
                elif obj == self.horizontal_gscene:
                    x, y, z = self.translate_gsceneCoord_to_3d('horizontal', gscene_x, gscene_y)
                self.update_cross(x, y, z)
                return True

            elif self.mode in ['midline', 'symmetric']:

                if obj == self.sagittal_gscene:
                    x, y, z = self.translate_gsceneCoord_to_3d('sagittal', gscene_x, gscene_y)
                elif obj == self.coronal_gscene:
                    x, y, z = self.translate_gsceneCoord_to_3d('coronal', gscene_x, gscene_y)
                elif obj == self.horizontal_gscene:
                    x, y, z = self.translate_gsceneCoord_to_3d('horizontal', gscene_x, gscene_y)

                if self.mode == 'midline':
                    self.recorded_points['midline'].append((x,y,z))
                    print 'midline', x,y,z

                elif self.mode == 'symmetric':
                    if not self.first_point_done:
                        self.first_point = (x,y,z)
                        print 'first point', self.first_point
                        self.first_point_done = True
                    else:
                        self.second_point = (x,y,z)
                        print 'second point', self.second_point
                        self.first_point_done = False
                        self.recorded_points['symmetric'].append((self.first_point, self.second_point))

                    # self.mode = None

                return True

        return False



if __name__ == "__main__":

    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Launch rectification GUI.')

    parser.add_argument("stack_name", type=str, help="stack name")
    args = parser.parse_args()

    from sys import argv, exit
    app = QApplication(argv)

    m = RectificationTool(stack=args.stack_name)

    # m.show()
    m.showMaximized()
    # m.raise_()
    exit(app.exec_())
