# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'input_selection_columnview.ui'
#
# Created: Thu Oct  9 06:14:37 2014
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

class Ui_InputSelectionDialog(object):
    def setupUi(self, InputSelectionDialog):
        InputSelectionDialog.setObjectName(_fromUtf8("InputSelectionDialog"))
        InputSelectionDialog.resize(924, 561)
        self.verticalLayout = QtGui.QVBoxLayout(InputSelectionDialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.StackSliceView = QtGui.QColumnView(InputSelectionDialog)
        self.StackSliceView.setObjectName(_fromUtf8("StackSliceView"))
        self.horizontalLayout_2.addWidget(self.StackSliceView)
        self.preview_pic = QtGui.QLabel(InputSelectionDialog)
        self.preview_pic.setObjectName(_fromUtf8("preview_pic"))
        self.horizontalLayout_2.addWidget(self.preview_pic)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(InputSelectionDialog)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.usernameEdit = QtGui.QLineEdit(InputSelectionDialog)
        self.usernameEdit.setObjectName(_fromUtf8("usernameEdit"))
        self.horizontalLayout.addWidget(self.usernameEdit)
        self.inputLoadButton = QtGui.QPushButton(InputSelectionDialog)
        self.inputLoadButton.setObjectName(_fromUtf8("inputLoadButton"))
        self.horizontalLayout.addWidget(self.inputLoadButton)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(InputSelectionDialog)
        QtCore.QMetaObject.connectSlotsByName(InputSelectionDialog)

    def retranslateUi(self, InputSelectionDialog):
        InputSelectionDialog.setWindowTitle(_translate("InputSelectionDialog", "Dialog", None))
        self.preview_pic.setText(_translate("InputSelectionDialog", "preview", None))
        self.label.setText(_translate("InputSelectionDialog", "Username", None))
        self.usernameEdit.setText(_translate("InputSelectionDialog", "anon", None))
        self.inputLoadButton.setText(_translate("InputSelectionDialog", "None", None))

