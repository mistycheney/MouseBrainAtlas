# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BrainLabelingGui_v8.ui'
#
# Created: Sun Feb  1 02:33:02 2015
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
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_2 = QtGui.QLabel(self.display_groupBox)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_3.addWidget(self.label_2)
        self.spOnOffSlider = QtGui.QSlider(self.display_groupBox)
        self.spOnOffSlider.setStyleSheet(_fromUtf8("QSlider {\n"
"min-width:80px;\n"
"min-height:27px;\n"
"max-width:80px;\n"
"max-height:27px;\n"
"}\n"
"QSlider::groove:horizontal {\n"
"background-image: url(images/slider_bg.png);\n"
"background-repeat: no-repeat;\n"
"background-position:center;\n"
"margin:0px;\n"
"border:0px;\n"
"padding:0px;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal {\n"
"background-image: url(images/slider_on.png);\n"
"background-repeat: no-repeat;\n"
"background-position:left;\n"
"background-origin:content;\n"
"margin:0px;\n"
"border:0px;\n"
"padding-left:0px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal {\n"
"background-image: url(images/slider_off.png);\n"
"background-repeat: no-repeat;\n"
"background-position:right;\n"
"background-origin:content;\n"
"margin:0px;\n"
"border:0px;\n"
"padding-right:0px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"background-image: url(images/slider_handle.png);\n"
"width:39px;\n"
"height:27px;\n"
"margin:0px;\n"
"border:0px;\n"
"padding:0px;\n"
"}\n"
"\n"
"QSlider::sub-page:horizontal:disabled {\n"
"background-image: url(images/slider_on_disabled.png);\n"
"background-repeat: no-repeat;\n"
"background-position:left;\n"
"background-origin:content;\n"
"margin:0px;\n"
"border:0px;\n"
"padding-left:0px;\n"
"}\n"
"\n"
"QSlider::add-page:horizontal:disabled {\n"
"background-image: url(images/slider_off_disabled.png);\n"
"background-repeat: no-repeat;\n"
"background-position:right;\n"
"background-origin:content;\n"
"margin:0px;\n"
"border:0px;\n"
"padding-right:0px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal:disabled {\n"
"background-image: url(images/slider_handle.png);\n"
"width:39px;\n"
"height:27px;\n"
"margin:0px;\n"
"border:0px;\n"
"padding:0px;\n"
"}\n"
"\n"
""))
        self.spOnOffSlider.setMaximum(1)
        self.spOnOffSlider.setSingleStep(1)
        self.spOnOffSlider.setTracking(False)
        self.spOnOffSlider.setOrientation(QtCore.Qt.Horizontal)
        self.spOnOffSlider.setObjectName(_fromUtf8("spOnOffSlider"))
        self.horizontalLayout_3.addWidget(self.spOnOffSlider)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.verticalLayout_3.addWidget(self.display_groupBox)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
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
        self.buttonParams = QtGui.QPushButton(self.centralwidget)
        self.buttonParams.setObjectName(_fromUtf8("buttonParams"))
        self.buttonsLayout.addWidget(self.buttonParams)
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
        self.textonmap_radioButton.setText(_translate("BrainLabelingGui", "textonmap (gabor, vq)", None))
        self.dirmap_radioButton.setText(_translate("BrainLabelingGui", "directionality (gabor, segm)", None))
        self.labeling_radioButton.setText(_translate("BrainLabelingGui", "labeling", None))
        self.label_2.setText(_translate("BrainLabelingGui", "superpixels (segm)", None))
        self.label.setText(_translate("BrainLabelingGui", "brush size", None))
        self.loadButton.setText(_translate("BrainLabelingGui", "Load", None))
        self.saveButton.setText(_translate("BrainLabelingGui", "Save", None))
        self.newLabelButton.setText(_translate("BrainLabelingGui", "Add New Label", None))
        self.buttonParams.setText(_translate("BrainLabelingGui", "Change Parameters", None))
        self.quitButton.setText(_translate("BrainLabelingGui", "Quit", None))
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

