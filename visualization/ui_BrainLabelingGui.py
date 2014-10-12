# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BrainLabelingGui.ui'
#
# Created: Thu Oct  9 10:23:05 2014
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

class Ui_BrainLabelingGui(object):
    def setupUi(self, BrainLabelingGui):
        BrainLabelingGui.setObjectName(_fromUtf8("BrainLabelingGui"))
        BrainLabelingGui.resize(965, 749)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(BrainLabelingGui.sizePolicy().hasHeightForWidth())
        BrainLabelingGui.setSizePolicy(sizePolicy)
        self.centralwidget = QtGui.QWidget(BrainLabelingGui)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.canvaswidget = MplWidget(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.canvaswidget.sizePolicy().hasHeightForWidth())
        self.canvaswidget.setSizePolicy(sizePolicy)
        self.canvaswidget.setObjectName(_fromUtf8("canvaswidget"))
        self.verticalLayout.addWidget(self.canvaswidget)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.labelsLayout = QtGui.QGridLayout()
        self.labelsLayout.setObjectName(_fromUtf8("labelsLayout"))
        self.horizontalLayout.addLayout(self.labelsLayout)
        self.buttonsLayout = QtGui.QVBoxLayout()
        self.buttonsLayout.setObjectName(_fromUtf8("buttonsLayout"))
        self.loadButton = QtGui.QPushButton(self.centralwidget)
        self.loadButton.setEnabled(True)
        self.loadButton.setCheckable(False)
        self.loadButton.setDefault(False)
        self.loadButton.setFlat(False)
        self.loadButton.setObjectName(_fromUtf8("loadButton"))
        self.buttonsLayout.addWidget(self.loadButton)
        self.saveButton = QtGui.QPushButton(self.centralwidget)
        self.saveButton.setObjectName(_fromUtf8("saveButton"))
        self.buttonsLayout.addWidget(self.saveButton)
        self.newLabelButton = QtGui.QPushButton(self.centralwidget)
        self.newLabelButton.setObjectName(_fromUtf8("newLabelButton"))
        self.buttonsLayout.addWidget(self.newLabelButton)
        self.sigboostButton = QtGui.QPushButton(self.centralwidget)
        self.sigboostButton.setObjectName(_fromUtf8("sigboostButton"))
        self.buttonsLayout.addWidget(self.sigboostButton)
        self.quitButton = QtGui.QPushButton(self.centralwidget)
        self.quitButton.setObjectName(_fromUtf8("quitButton"))
        self.buttonsLayout.addWidget(self.quitButton)
        self.horizontalLayout.addLayout(self.buttonsLayout)
        self.verticalLayout.addLayout(self.horizontalLayout)
        BrainLabelingGui.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(BrainLabelingGui)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        BrainLabelingGui.setStatusBar(self.statusbar)
        self.menubar = QtGui.QMenuBar(BrainLabelingGui)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 965, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        BrainLabelingGui.setMenuBar(self.menubar)

        self.retranslateUi(BrainLabelingGui)
        QtCore.QMetaObject.connectSlotsByName(BrainLabelingGui)

    def retranslateUi(self, BrainLabelingGui):
        BrainLabelingGui.setWindowTitle(_translate("BrainLabelingGui", "BrainLabelingGui", None))
        self.loadButton.setText(_translate("BrainLabelingGui", "Load", None))
        self.saveButton.setText(_translate("BrainLabelingGui", "Save", None))
        self.newLabelButton.setText(_translate("BrainLabelingGui", "Add Label", None))
        self.sigboostButton.setText(_translate("BrainLabelingGui", "SigBoost", None))
        self.quitButton.setText(_translate("BrainLabelingGui", "Quit", None))

from mplwidget import MplWidget
