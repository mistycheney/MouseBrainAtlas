# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BrainLabelingGui_v13.ui'
#
# Created: Thu Mar 17 15:55:07 2016
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
        BrainLabelingGui.resize(1314, 1066)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(BrainLabelingGui.sizePolicy().hasHeightForWidth())
        BrainLabelingGui.setSizePolicy(sizePolicy)
        self.centralwidget = QtGui.QWidget(BrainLabelingGui)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_6 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.thumbnail_list = QtGui.QListWidget(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thumbnail_list.sizePolicy().hasHeightForWidth())
        self.thumbnail_list.setSizePolicy(sizePolicy)
        self.thumbnail_list.setMinimumSize(QtCore.QSize(256, 773))
        self.thumbnail_list.setObjectName(_fromUtf8("thumbnail_list"))
        self.verticalLayout_2.addWidget(self.thumbnail_list)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_3.addWidget(self.label)
        self.lineEdit_username = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_username.setObjectName(_fromUtf8("lineEdit_username"))
        self.horizontalLayout_3.addWidget(self.lineEdit_username)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.button_saveLabeling = QtGui.QPushButton(self.centralwidget)
        self.button_saveLabeling.setObjectName(_fromUtf8("button_saveLabeling"))
        self.horizontalLayout_2.addWidget(self.button_saveLabeling)
        self.button_quit = QtGui.QPushButton(self.centralwidget)
        self.button_quit.setObjectName(_fromUtf8("button_quit"))
        self.horizontalLayout_2.addWidget(self.button_quit)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.splitter = QtGui.QSplitter(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.widget = QtGui.QWidget(self.splitter)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.widget)
        self.verticalLayout_3.setMargin(0)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.section3_gview = QtGui.QGraphicsView(self.widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.section3_gview.sizePolicy().hasHeightForWidth())
        self.section3_gview.setSizePolicy(sizePolicy)
        self.section3_gview.setMinimumSize(QtCore.QSize(0, 0))
        self.section3_gview.setObjectName(_fromUtf8("section3_gview"))
        self.verticalLayout_3.addWidget(self.section3_gview)
        self.button_loadLabelingSec1 = QtGui.QPushButton(self.widget)
        self.button_loadLabelingSec1.setObjectName(_fromUtf8("button_loadLabelingSec1"))
        self.verticalLayout_3.addWidget(self.button_loadLabelingSec1)
        self.widget1 = QtGui.QWidget(self.splitter)
        self.widget1.setObjectName(_fromUtf8("widget1"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.widget1)
        self.verticalLayout_4.setMargin(0)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.section1_gview = QtGui.QGraphicsView(self.widget1)
        self.section1_gview.setEnabled(True)
        self.section1_gview.setMinimumSize(QtCore.QSize(0, 0))
        self.section1_gview.setObjectName(_fromUtf8("section1_gview"))
        self.verticalLayout_4.addWidget(self.section1_gview)
        self.button_loadLabelingSec2 = QtGui.QPushButton(self.widget1)
        self.button_loadLabelingSec2.setObjectName(_fromUtf8("button_loadLabelingSec2"))
        self.verticalLayout_4.addWidget(self.button_loadLabelingSec2)
        self.widget2 = QtGui.QWidget(self.splitter)
        self.widget2.setObjectName(_fromUtf8("widget2"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.widget2)
        self.verticalLayout_5.setMargin(0)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.section2_gview = QtGui.QGraphicsView(self.widget2)
        self.section2_gview.setMinimumSize(QtCore.QSize(0, 0))
        self.section2_gview.setObjectName(_fromUtf8("section2_gview"))
        self.verticalLayout_5.addWidget(self.section2_gview)
        self.button_loadLabelingSec3 = QtGui.QPushButton(self.widget2)
        self.button_loadLabelingSec3.setObjectName(_fromUtf8("button_loadLabelingSec3"))
        self.verticalLayout_5.addWidget(self.button_loadLabelingSec3)
        self.horizontalLayout.addWidget(self.splitter)
        self.verticalLayout_6.addLayout(self.horizontalLayout)
        BrainLabelingGui.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(BrainLabelingGui)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        BrainLabelingGui.setStatusBar(self.statusbar)
        self.menubar = QtGui.QMenuBar(BrainLabelingGui)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1314, 25))
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
        self.label.setText(_translate("BrainLabelingGui", "Username:", None))
        self.lineEdit_username.setText(_translate("BrainLabelingGui", "anon", None))
        self.button_saveLabeling.setText(_translate("BrainLabelingGui", "Save", None))
        self.button_quit.setText(_translate("BrainLabelingGui", "Quit", None))
        self.button_loadLabelingSec1.setText(_translate("BrainLabelingGui", "Load", None))
        self.button_loadLabelingSec2.setText(_translate("BrainLabelingGui", "Load", None))
        self.button_loadLabelingSec3.setText(_translate("BrainLabelingGui", "Load", None))
        self.toolBar.setWindowTitle(_translate("BrainLabelingGui", "toolBar", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    BrainLabelingGui = QtGui.QMainWindow()
    ui = Ui_BrainLabelingGui()
    ui.setupUi(BrainLabelingGui)
    BrainLabelingGui.show()
    sys.exit(app.exec_())

