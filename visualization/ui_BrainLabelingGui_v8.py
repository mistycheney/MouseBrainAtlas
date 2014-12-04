# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BrainLabelingGui_v8.ui'
#
# Created: Thu Dec  4 06:16:02 2014
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
        self.canvaswidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.canvaswidget.setObjectName(_fromUtf8("canvaswidget"))
        self.verticalLayout.addWidget(self.canvaswidget)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.labelsLayout = QtGui.QGridLayout()
        self.labelsLayout.setObjectName(_fromUtf8("labelsLayout"))
        self.horizontalLayout.addLayout(self.labelsLayout)
        self.display_groupBox = QtGui.QGroupBox(self.centralwidget)
        self.display_groupBox.setObjectName(_fromUtf8("display_groupBox"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.display_groupBox)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.img_radioButton = QtGui.QRadioButton(self.display_groupBox)
        self.img_radioButton.setObjectName(_fromUtf8("img_radioButton"))
        self.verticalLayout_2.addWidget(self.img_radioButton)
        self.imgSeg_radioButton = QtGui.QRadioButton(self.display_groupBox)
        self.imgSeg_radioButton.setObjectName(_fromUtf8("imgSeg_radioButton"))
        self.verticalLayout_2.addWidget(self.imgSeg_radioButton)
        self.textonmap_radioButton = QtGui.QRadioButton(self.display_groupBox)
        self.textonmap_radioButton.setObjectName(_fromUtf8("textonmap_radioButton"))
        self.verticalLayout_2.addWidget(self.textonmap_radioButton)
        self.dirmap_radioButton = QtGui.QRadioButton(self.display_groupBox)
        self.dirmap_radioButton.setObjectName(_fromUtf8("dirmap_radioButton"))
        self.verticalLayout_2.addWidget(self.dirmap_radioButton)
        self.labeling_radioButton = QtGui.QRadioButton(self.display_groupBox)
        self.labeling_radioButton.setObjectName(_fromUtf8("labeling_radioButton"))
        self.verticalLayout_2.addWidget(self.labeling_radioButton)
        self.horizontalLayout.addWidget(self.display_groupBox)
        self.buttonsLayout = QtGui.QVBoxLayout()
        self.buttonsLayout.setObjectName(_fromUtf8("buttonsLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_2.addWidget(self.label)
        self.brushSizeSlider = QtGui.QSlider(self.centralwidget)
        self.brushSizeSlider.setMaximum(50)
        self.brushSizeSlider.setProperty("value", 20)
        self.brushSizeSlider.setOrientation(QtCore.Qt.Horizontal)
        self.brushSizeSlider.setInvertedControls(False)
        self.brushSizeSlider.setTickPosition(QtGui.QSlider.NoTicks)
        self.brushSizeSlider.setObjectName(_fromUtf8("brushSizeSlider"))
        self.horizontalLayout_2.addWidget(self.brushSizeSlider)
        self.brushSizeEdit = QtGui.QLineEdit(self.centralwidget)
        self.brushSizeEdit.setText(_fromUtf8(""))
        self.brushSizeEdit.setObjectName(_fromUtf8("brushSizeEdit"))
        self.horizontalLayout_2.addWidget(self.brushSizeEdit)
        self.buttonsLayout.addLayout(self.horizontalLayout_2)
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
        self.toolBar = QtGui.QToolBar(BrainLabelingGui)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        BrainLabelingGui.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.toolBar.addSeparator()

        self.retranslateUi(BrainLabelingGui)
        QtCore.QMetaObject.connectSlotsByName(BrainLabelingGui)

    def retranslateUi(self, BrainLabelingGui):
        BrainLabelingGui.setWindowTitle(_translate("BrainLabelingGui", "BrainLabelingGui", None))
        self.display_groupBox.setTitle(_translate("BrainLabelingGui", "Display", None))
        self.img_radioButton.setText(_translate("BrainLabelingGui", "image", None))
        self.imgSeg_radioButton.setText(_translate("BrainLabelingGui", "image + segmentation", None))
        self.textonmap_radioButton.setText(_translate("BrainLabelingGui", "textonmap", None))
        self.dirmap_radioButton.setText(_translate("BrainLabelingGui", "directionality", None))
        self.labeling_radioButton.setText(_translate("BrainLabelingGui", "labeling", None))
        self.label.setText(_translate("BrainLabelingGui", "brush size", None))
        self.loadButton.setText(_translate("BrainLabelingGui", "Load", None))
        self.saveButton.setText(_translate("BrainLabelingGui", "Save", None))
        self.newLabelButton.setText(_translate("BrainLabelingGui", "Add New Label", None))
        self.quitButton.setText(_translate("BrainLabelingGui", "Quit", None))
        self.toolBar.setWindowTitle(_translate("BrainLabelingGui", "toolBar", None))

from mplwidget import MplWidget
