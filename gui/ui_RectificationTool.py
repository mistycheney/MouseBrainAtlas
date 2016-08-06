# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RectificationTool.ui'
#
# Created: Tue Aug  2 18:55:31 2016
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
        RectificationGUI.resize(1335, 1029)
        self.centralwidget = QtGui.QWidget(RectificationGUI)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.coronal_gview = QtGui.QGraphicsView(self.centralwidget)
        self.coronal_gview.setGeometry(QtCore.QRect(520, 10, 461, 401))
        self.coronal_gview.setObjectName(_fromUtf8("coronal_gview"))
        self.horizontal_gview = QtGui.QGraphicsView(self.centralwidget)
        self.horizontal_gview.setGeometry(QtCore.QRect(20, 420, 491, 401))
        self.horizontal_gview.setObjectName(_fromUtf8("horizontal_gview"))
        self.sagittal_gview = QtGui.QGraphicsView(self.centralwidget)
        self.sagittal_gview.setGeometry(QtCore.QRect(20, 10, 491, 401))
        self.sagittal_gview.setObjectName(_fromUtf8("sagittal_gview"))
        RectificationGUI.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(RectificationGUI)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1335, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        RectificationGUI.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(RectificationGUI)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        RectificationGUI.setStatusBar(self.statusbar)

        self.retranslateUi(RectificationGUI)
        QtCore.QMetaObject.connectSlotsByName(RectificationGUI)

    def retranslateUi(self, RectificationGUI):
        RectificationGUI.setWindowTitle(_translate("RectificationGUI", "MainWindow", None))
