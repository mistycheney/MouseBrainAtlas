# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DataManager.ui'
#
# Created: Thu Oct  9 06:55:25 2014
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

class Ui_DataManager(object):
    def setupUi(self, DataManager):
        DataManager.setObjectName(_fromUtf8("DataManager"))
        DataManager.resize(1024, 600)
        self.centralwidget = QtGui.QWidget(DataManager)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.StackSliceView = QtGui.QColumnView(self.centralwidget)
        self.StackSliceView.setObjectName(_fromUtf8("StackSliceView"))
        self.horizontalLayout_2.addWidget(self.StackSliceView)
        self.preview_pic = QtGui.QLabel(self.centralwidget)
        self.preview_pic.setObjectName(_fromUtf8("preview_pic"))
        self.horizontalLayout_2.addWidget(self.preview_pic)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.usernameEdit = QtGui.QLineEdit(self.centralwidget)
        self.usernameEdit.setObjectName(_fromUtf8("usernameEdit"))
        self.horizontalLayout.addWidget(self.usernameEdit)
        self.inputLoadButton = QtGui.QPushButton(self.centralwidget)
        self.inputLoadButton.setObjectName(_fromUtf8("inputLoadButton"))
        self.horizontalLayout.addWidget(self.inputLoadButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        DataManager.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(DataManager)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        DataManager.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(DataManager)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        DataManager.setStatusBar(self.statusbar)

        self.retranslateUi(DataManager)
        QtCore.QMetaObject.connectSlotsByName(DataManager)

    def retranslateUi(self, DataManager):
        DataManager.setWindowTitle(_translate("DataManager", "MainWindow", None))
        self.preview_pic.setText(_translate("DataManager", "preview", None))
        self.label.setText(_translate("DataManager", "Username", None))
        self.usernameEdit.setText(_translate("DataManager", "anon", None))
        self.inputLoadButton.setText(_translate("DataManager", "None", None))

