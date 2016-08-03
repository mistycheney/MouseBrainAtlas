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

PEN_WIDTH = 2

red_pen = QPen(Qt.red)
red_pen.setWidth(PEN_WIDTH)

# Use the third method in http://pyqt.sourceforge.net/Docs/PyQt4/designer.html
class RectificationTool(QMainWindow, Ui_RectificationGUI):
    def __init__(self, parent=None, stack=None):
        """
        Initialization of RectificationTool.
        """
        # self.app = QApplication(sys.argv)
        QMainWindow.__init__(self, parent)

        self.setupUi(self)

        # self.sagittal_gscene = QGraphicsScene(self.sagittal_gview)

        self.stack = stack

        self.coronal_gscene = QGraphicsScene()
        self.coronal_gview.setScene(self.coronal_gscene)

        self.horizontal_gscene = QGraphicsScene()
        self.horizontal_gview.setScene(self.horizontal_gscene)

        self.sagittal_gscene = QGraphicsScene()
        self.sagittal_gview.setScene(self.sagittal_gscene)

        # self.sagittal_gview.update(0, 0, self.sagittal_gview.width(), self.sagittal_gview.height())

        # self.sagittal_gview.show()

        self.coronal_gscene.installEventFilter(self)
        self.horizontal_gscene.installEventFilter(self)
        self.coronal_gview.viewport().installEventFilter(self)
        self.horizontal_gview.viewport().installEventFilter(self)

        # self.sagittal_gview.viewport().installEventFilter(self)

        self.sagittal_pixmapItem = QGraphicsPixmapItem()
        self.sagittal_gscene.addItem(self.sagittal_pixmapItem)

        self.coronal_pixmapItem = QGraphicsPixmapItem()
        self.coronal_gscene.addItem(self.coronal_pixmapItem)

        self.horizontal_pixmapItem = QGraphicsPixmapItem()
        self.horizontal_gscene.addItem(self.horizontal_pixmapItem)

        self.coronal_hline = QGraphicsLineItem()
        self.coronal_hline.setPen(red_pen)
        # self.coronal_hline.setZValue(0)
        self.coronal_vline = QGraphicsLineItem()
        self.coronal_vline.setPen(red_pen)
        # self.coronal_vline.setZValue(0)
        self.horizontal_hline = QGraphicsLineItem()
        self.horizontal_hline.setPen(red_pen)
        # self.horizontal_hline.setZValue(0)
        self.horizontal_vline = QGraphicsLineItem()
        self.horizontal_vline.setPen(red_pen)
        # self.horizontal_vline.setZValue(0)

        self.coronal_gscene.addItem(self.coronal_hline)
        self.coronal_gscene.addItem(self.coronal_vline)
        self.horizontal_gscene.addItem(self.horizontal_hline)
        self.horizontal_gscene.addItem(self.horizontal_vline)

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

        self.button_sameX.clicked.connect(self.sameX_clicked)
        self.button_sameY.clicked.connect(self.sameY_clicked)
        self.button_sameZ.clicked.connect(self.sameZ_clicked)

        self.first_point_done = False
        self.mode = None
        self.recorded_pairs = {'sameX': [], 'sameY': [], 'sameZ': []}

        self.update_cross(0,0,0)

    def update_crosslines(self):
        print self.cross_x, self.cross_y, self.cross_z
        self.coronal_hline.setLine(0, self.cross_y, self.z_dim-1, self.cross_y)
        self.coronal_vline.setLine(self.z_dim-1-self.cross_z, 0, self.z_dim-1-self.cross_z, self.y_dim-1)
        self.horizontal_hline.setLine(0, self.cross_x, self.z_dim-1, self.cross_x)
        self.horizontal_vline.setLine(self.z_dim-1-self.cross_z, 0, self.z_dim-1-self.cross_z, self.x_dim-1)
        # self.coronal_gscene.update(0, self.coronal_gscene.width(), 0, self.coronal_gscene.height())
        # self.horizontal_gscene.update(0, self.horizontal_gscene.width(), 0, self.horizontal_gscene.height())
        # self.coronal_gview.update()
        # self.horizontal_gview.update()

    def sameX_clicked(self):
        self.mode = 'sameX'

    def sameY_clicked(self):
        self.mode = 'sameY'

    def sameZ_clicked(self):
        self.mode = 'sameZ'

    def update_gscenes(self, which):

        if which == 'y':
            horizontal_fn = '/home/yuncong/CSHL_volumes_resection/horizontal/%(stack)s_thumbnailVolume_horizontal_%(y)03d.tif' % {'stack': self.stack, 'y': self.cross_y}
            self.horizontal_pixmap = QPixmap(horizontal_fn)
            self.horizontal_pixmapItem.setPixmap(self.horizontal_pixmap)

            self.x_dim = self.horizontal_pixmap.height()
            self.z_dim = self.horizontal_pixmap.width()

        elif which == 'x':
            coronal_fn = '/home/yuncong/CSHL_volumes_resection/coronal/%(stack)s_thumbnailVolume_coronal_%(x)03d.tif' % {'stack': self.stack, 'x': self.cross_x}
            self.coronal_pixmap = QPixmap(coronal_fn)
            self.coronal_pixmapItem.setPixmap(self.coronal_pixmap)

            self.z_dim = self.coronal_pixmap.width()
            self.y_dim = self.coronal_pixmap.height()

    def update_cross(self, x=None, y=None, z=None):

        if y is not None:
            self.cross_y = y
            self.update_gscenes('y')
        if x is not None:
            self.cross_x = x
            self.update_gscenes('x')
        if z is not None:
            self.cross_z = z
            self.update_gscenes('z')

        self.update_crosslines()

    def eventFilter(self, obj, event):

        event_type = event.type()

        if event_type == QEvent.GraphicsSceneWheel or event_type == QEvent.Wheel:

            if event.delta() < 0: # negative means towards user
                if obj == self.sagittal_gscene or obj == self.sagittal_gview.viewport():
                    self.update_cross(z=min(self.cross_z+1, self.z_dim-1))
                elif obj == self.coronal_gscene or obj == self.coronal_gview.viewport():
                    self.update_cross(x=min(self.cross_x+1, self.x_dim-1))
                elif obj == self.horizontal_gscene or obj == self.horizontal_gview.viewport():
                    self.update_cross(y=min(self.cross_y+1, self.y_dim-1))
            else: # scroll away from user
                if obj == self.sagittal_gscene or obj == self.sagittal_gview.viewport():
                    self.update_cross(z=max(self.cross_z-1, 0))
                elif obj == self.coronal_gscene or obj == self.coronal_gview.viewport():
                    self.update_cross(x=max(self.cross_x-1, 0))
                elif obj == self.horizontal_gscene or obj == self.horizontal_gview.viewport():
                    self.update_cross(y=max(self.cross_y-1, 0))

            return True

        elif event_type == QEvent.GraphicsSceneMousePress:

            gscene_x = event.scenePos().x()
            gscene_y = event.scenePos().y()

            if self.mode is None:

                if obj == self.sagittal_gscene:
                    pass
                elif obj == self.coronal_gscene:
                    self.update_cross(z=self.z_dim - 1 - gscene_x, y=gscene_y)
                elif obj == self.horizontal_gscene:
                    self.update_cross(x=gscene_y, z=self.z_dim - 1 - gscene_x)
                return True

            else:

                if obj == self.sagittal_gscene:
                    return True
                elif obj == self.coronal_gscene:
                    x = self.cross_x
                    y = gscene_y
                    z = -gscene_x
                    if not self.first_point_done:
                        self.first_point = (x,y,z)
                        print 'first point', self.first_point
                        self.first_point_done = True
                    else:
                        self.second_point = (x,y,z)
                        print 'second point', self.second_point
                        self.first_point_done = False
                        self.recorded_pairs[self.mode].append((self.first_point, self.second_point))
                        self.mode = None
                    return True
                elif obj == self.horizontal_gscene:
                    x = -gscene_y
                    y = self.cross_y
                    z = -gscene_x
                    if not self.first_point_done:
                        self.first_point = (x,y,z)
                        print 'first point', self.first_point
                        self.first_point_done = True
                    else:
                        self.second_point = (x,y,z)
                        print 'second point', self.second_point
                        self.first_point_done = False
                        self.recorded_pairs[self.mode].append((self.first_point, self.second_point))
                        self.mode = None
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

    stack = args.stack_name
    m = RectificationTool(stack=stack)

    m.show()
    # m.showMaximized()
    # m.raise_()
    exit(app.exec_())
