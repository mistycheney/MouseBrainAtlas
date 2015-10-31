# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BrainLabelingGui_v10.ui'
#
# Created: Fri Oct 30 15:58:56 2015
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
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.groupBox = QtGui.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.groupBox)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.radioButton_globalProposal = QtGui.QRadioButton(self.groupBox)
        self.radioButton_globalProposal.setObjectName(_fromUtf8("radioButton_globalProposal"))
        self.verticalLayout_4.addWidget(self.radioButton_globalProposal)
        self.radioButton_localProposal = QtGui.QRadioButton(self.groupBox)
        self.radioButton_localProposal.setObjectName(_fromUtf8("radioButton_localProposal"))
        self.verticalLayout_4.addWidget(self.radioButton_localProposal)
        self.horizontalLayout.addWidget(self.groupBox)
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.display_groupBox = QtGui.QGroupBox(self.centralwidget)
        self.display_groupBox.setObjectName(_fromUtf8("display_groupBox"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.display_groupBox)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.img_radioButton = QtGui.QRadioButton(self.display_groupBox)
        self.img_radioButton.setObjectName(_fromUtf8("img_radioButton"))
        self.verticalLayout_2.addWidget(self.img_radioButton)
        self.textonmap_radioButton = QtGui.QRadioButton(self.display_groupBox)
        self.textonmap_radioButton.setObjectName(_fromUtf8("textonmap_radioButton"))
        self.verticalLayout_2.addWidget(self.textonmap_radioButton)
        self.dirmap_radioButton = QtGui.QRadioButton(self.display_groupBox)
        self.dirmap_radioButton.setObjectName(_fromUtf8("dirmap_radioButton"))
        self.verticalLayout_2.addWidget(self.dirmap_radioButton)
        self.labeling_radioButton = QtGui.QRadioButton(self.display_groupBox)
        self.labeling_radioButton.setObjectName(_fromUtf8("labeling_radioButton"))
        self.verticalLayout_2.addWidget(self.labeling_radioButton)
        self.buttonSpOnOff = QtGui.QPushButton(self.display_groupBox)
        self.buttonSpOnOff.setObjectName(_fromUtf8("buttonSpOnOff"))
        self.verticalLayout_2.addWidget(self.buttonSpOnOff)
        self.verticalLayout_3.addWidget(self.display_groupBox)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.buttonsLayout = QtGui.QVBoxLayout()
        self.buttonsLayout.setObjectName(_fromUtf8("buttonsLayout"))
        self.button_loadLabeling = QtGui.QPushButton(self.centralwidget)
        self.button_loadLabeling.setObjectName(_fromUtf8("button_loadLabeling"))
        self.buttonsLayout.addWidget(self.button_loadLabeling)
        self.button_saveLabeling = QtGui.QPushButton(self.centralwidget)
        self.button_saveLabeling.setObjectName(_fromUtf8("button_saveLabeling"))
        self.buttonsLayout.addWidget(self.button_saveLabeling)
        self.buttonParams = QtGui.QPushButton(self.centralwidget)
        self.buttonParams.setObjectName(_fromUtf8("buttonParams"))
        self.buttonsLayout.addWidget(self.buttonParams)
        self.button_quit = QtGui.QPushButton(self.centralwidget)
        self.button_quit.setObjectName(_fromUtf8("button_quit"))
        self.buttonsLayout.addWidget(self.button_quit)
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
        self.groupBox.setTitle(_translate("BrainLabelingGui", "Proposal Source", None))
        self.radioButton_globalProposal.setText(_translate("BrainLabelingGui", "Global Proposals", None))
        self.radioButton_localProposal.setText(_translate("BrainLabelingGui", "Local Proposals", None))
        self.display_groupBox.setTitle(_translate("BrainLabelingGui", "Display", None))
        self.img_radioButton.setText(_translate("BrainLabelingGui", "image", None))
        self.textonmap_radioButton.setText(_translate("BrainLabelingGui", "textonmap (gabor, vq)", None))
        self.dirmap_radioButton.setText(_translate("BrainLabelingGui", "directionality (gabor, segm)", None))
        self.labeling_radioButton.setText(_translate("BrainLabelingGui", "labeling", None))
        self.buttonSpOnOff.setText(_translate("BrainLabelingGui", "Turn Superpixels On", None))
        self.button_loadLabeling.setText(_translate("BrainLabelingGui", "Load", None))
        self.button_saveLabeling.setText(_translate("BrainLabelingGui", "Save", None))
        self.buttonParams.setText(_translate("BrainLabelingGui", "Change Parameters", None))
        self.button_quit.setText(_translate("BrainLabelingGui", "Quit", None))
        self.toolBar.setWindowTitle(_translate("BrainLabelingGui", "toolBar", None))

from mplwidget import MplWidget

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    BrainLabelingGui = QtGui.QMainWindow()
    ui = Ui_BrainLabelingGui()
    ui.setupUi(BrainLabelingGui)
    BrainLabelingGui.show()
    sys.exit(app.exec_())

