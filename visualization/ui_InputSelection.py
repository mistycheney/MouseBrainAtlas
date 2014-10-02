# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'input_selection.ui'
#
# Created: Thu Oct  2 00:39:12 2014
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
        InputSelectionDialog.resize(916, 575)
        self.StackSliceView = QtGui.QColumnView(InputSelectionDialog)
        self.StackSliceView.setGeometry(QtCore.QRect(10, 50, 501, 451))
        self.StackSliceView.setObjectName(_fromUtf8("StackSliceView"))
        self.ResolutionList = QtGui.QListWidget(InputSelectionDialog)
        self.ResolutionList.setGeometry(QtCore.QRect(520, 50, 151, 192))
        self.ResolutionList.setObjectName(_fromUtf8("ResolutionList"))
        self.ParamList = QtGui.QListWidget(InputSelectionDialog)
        self.ParamList.setGeometry(QtCore.QRect(680, 50, 141, 192))
        self.ParamList.setObjectName(_fromUtf8("ParamList"))
        self.label = QtGui.QLabel(InputSelectionDialog)
        self.label.setGeometry(QtCore.QRect(10, 20, 91, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(InputSelectionDialog)
        self.label_2.setGeometry(QtCore.QRect(520, 20, 91, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(InputSelectionDialog)
        self.label_3.setGeometry(QtCore.QRect(680, 20, 111, 17))
        self.label_3.setObjectName(_fromUtf8("label_3"))

        self.retranslateUi(InputSelectionDialog)
        QtCore.QMetaObject.connectSlotsByName(InputSelectionDialog)

    def retranslateUi(self, InputSelectionDialog):
        InputSelectionDialog.setWindowTitle(_translate("InputSelectionDialog", "Dialog", None))
        self.label.setText(_translate("InputSelectionDialog", "Stack/Slice", None))
        self.label_2.setText(_translate("InputSelectionDialog", "Resolution", None))
        self.label_3.setText(_translate("InputSelectionDialog", "Parameter Set", None))

