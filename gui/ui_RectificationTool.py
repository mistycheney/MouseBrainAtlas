# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RectificationTool.ui'
#
# Created: Wed Aug  3 19:11:39 2016
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_RectificationGUI(object):
    def setupUi(self, RectificationGUI):
        RectificationGUI.setObjectName(_fromUtf8("RectificationGUI"))
        RectificationGUI.resize(1521, 1113)
        self.centralwidget = QtGui.QWidget(RectificationGUI)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.coronal_gview = QtGui.QGraphicsView(self.centralwidget)
        self.coronal_gview.setGeometry(QtCore.QRect(870, 10, 631, 511))
        self.coronal_gview.setObjectName(_fromUtf8("coronal_gview"))
        self.horizontal_gview = QtGui.QGraphicsView(self.centralwidget)
        self.horizontal_gview.setGeometry(QtCore.QRect(10, 530, 841, 521))
        self.horizontal_gview.setObjectName(_fromUtf8("horizontal_gview"))
        self.sagittal_gview = QtGui.QGraphicsView(self.centralwidget)
        self.sagittal_gview.setGeometry(QtCore.QRect(10, 10, 841, 511))
        self.sagittal_gview.setObjectName(_fromUtf8("sagittal_gview"))
        self.button_sameX = QtGui.QPushButton(self.centralwidget)
        self.button_sameX.setGeometry(QtCore.QRect(930, 860, 98, 27))
        self.button_sameX.setObjectName(_fromUtf8("button_sameX"))
        self.button_sameY = QtGui.QPushButton(self.centralwidget)
        self.button_sameY.setGeometry(QtCore.QRect(930, 890, 98, 27))
        self.button_sameY.setObjectName(_fromUtf8("button_sameY"))
        self.button_sameZ = QtGui.QPushButton(self.centralwidget)
        self.button_sameZ.setGeometry(QtCore.QRect(930, 920, 98, 27))
        self.button_sameZ.setObjectName(_fromUtf8("button_sameZ"))
        self.button_done = QtGui.QPushButton(self.centralwidget)
        self.button_done.setGeometry(QtCore.QRect(930, 970, 98, 27))
        self.button_done.setObjectName(_fromUtf8("button_done"))
        self.slider_hxy = QtGui.QSlider(self.centralwidget)
        self.slider_hxy.setGeometry(QtCore.QRect(990, 570, 160, 29))
        self.slider_hxy.setMinimum(-100)
        self.slider_hxy.setMaximum(100)
        self.slider_hxy.setTracking(True)
        self.slider_hxy.setOrientation(QtCore.Qt.Horizontal)
        self.slider_hxy.setInvertedAppearance(False)
        self.slider_hxy.setInvertedControls(False)
        self.slider_hxy.setTickPosition(QtGui.QSlider.TicksAbove)
        self.slider_hxy.setObjectName(_fromUtf8("slider_hxy"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(910, 580, 66, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(910, 610, 66, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.slider_hyx = QtGui.QSlider(self.centralwidget)
        self.slider_hyx.setGeometry(QtCore.QRect(990, 600, 160, 29))
        self.slider_hyx.setMinimum(-100)
        self.slider_hyx.setMaximum(100)
        self.slider_hyx.setOrientation(QtCore.Qt.Horizontal)
        self.slider_hyx.setObjectName(_fromUtf8("slider_hyx"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(910, 640, 66, 17))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.slider_hzx = QtGui.QSlider(self.centralwidget)
        self.slider_hzx.setGeometry(QtCore.QRect(990, 630, 160, 29))
        self.slider_hzx.setMinimum(-100)
        self.slider_hzx.setMaximum(100)
        self.slider_hzx.setOrientation(QtCore.Qt.Horizontal)
        self.slider_hzx.setObjectName(_fromUtf8("slider_hzx"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(1180, 580, 66, 17))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.slider_hxz = QtGui.QSlider(self.centralwidget)
        self.slider_hxz.setGeometry(QtCore.QRect(1260, 570, 160, 29))
        self.slider_hxz.setMinimum(-100)
        self.slider_hxz.setMaximum(100)
        self.slider_hxz.setOrientation(QtCore.Qt.Horizontal)
        self.slider_hxz.setObjectName(_fromUtf8("slider_hxz"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(1180, 610, 66, 17))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.slider_hyz = QtGui.QSlider(self.centralwidget)
        self.slider_hyz.setGeometry(QtCore.QRect(1260, 600, 160, 29))
        self.slider_hyz.setMinimum(-100)
        self.slider_hyz.setMaximum(100)
        self.slider_hyz.setOrientation(QtCore.Qt.Horizontal)
        self.slider_hyz.setObjectName(_fromUtf8("slider_hyz"))
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(1180, 640, 66, 17))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.slider_hzy = QtGui.QSlider(self.centralwidget)
        self.slider_hzy.setGeometry(QtCore.QRect(1260, 630, 160, 29))
        self.slider_hzy.setMinimum(-100)
        self.slider_hzy.setMaximum(100)
        self.slider_hzy.setOrientation(QtCore.Qt.Horizontal)
        self.slider_hzy.setObjectName(_fromUtf8("slider_hzy"))
        RectificationGUI.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(RectificationGUI)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1521, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        RectificationGUI.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(RectificationGUI)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        RectificationGUI.setStatusBar(self.statusbar)

        self.retranslateUi(RectificationGUI)
        QtCore.QMetaObject.connectSlotsByName(RectificationGUI)

    def retranslateUi(self, RectificationGUI):
        RectificationGUI.setWindowTitle(_translate("RectificationGUI", "MainWindow", None))
        self.button_sameX.setText(_translate("RectificationGUI", "Same x", None))
        self.button_sameY.setText(_translate("RectificationGUI", "Same y", None))
        self.button_sameZ.setText(_translate("RectificationGUI", "Same z", None))
        self.button_done.setText(_translate("RectificationGUI", "Complete", None))
        self.label.setText(_translate("RectificationGUI", "h_xy", None))
        self.label_2.setText(_translate("RectificationGUI", "h_yx", None))
        self.label_3.setText(_translate("RectificationGUI", "h_zx", None))
        self.label_4.setText(_translate("RectificationGUI", "h_xz", None))
        self.label_5.setText(_translate("RectificationGUI", "h_yz", None))
        self.label_6.setText(_translate("RectificationGUI", "h_zy", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    RectificationGUI = QtGui.QMainWindow()
    ui = Ui_RectificationGUI()
    ui.setupUi(RectificationGUI)
    RectificationGUI.show()
    sys.exit(app.exec_())

