# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'input_selection_multiplelists.ui'
#
# Created: Mon Oct  6 22:11:44 2014
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
        self.label = QtGui.QLabel(InputSelectionDialog)
        self.label.setGeometry(QtCore.QRect(10, 520, 71, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.usernameEdit = QtGui.QLineEdit(InputSelectionDialog)
        self.usernameEdit.setGeometry(QtCore.QRect(90, 520, 113, 27))
        self.usernameEdit.setObjectName(_fromUtf8("usernameEdit"))
        self.layoutWidget = QtGui.QWidget(InputSelectionDialog)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 40, 861, 451))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_2 = QtGui.QLabel(self.layoutWidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_3 = QtGui.QLabel(self.layoutWidget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 0, 1, 1, 1)
        self.label_4 = QtGui.QLabel(self.layoutWidget)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 0, 2, 1, 1)
        self.label_5 = QtGui.QLabel(self.layoutWidget)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout.addWidget(self.label_5, 0, 3, 1, 1)
        self.label_6 = QtGui.QLabel(self.layoutWidget)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout.addWidget(self.label_6, 0, 4, 1, 1)
        self.StackView = QtGui.QListView(self.layoutWidget)
        self.StackView.setObjectName(_fromUtf8("StackView"))
        self.gridLayout.addWidget(self.StackView, 1, 0, 1, 1)
        self.ResolView = QtGui.QListView(self.layoutWidget)
        self.ResolView.setObjectName(_fromUtf8("ResolView"))
        self.gridLayout.addWidget(self.ResolView, 1, 1, 1, 1)
        self.SliceView = QtGui.QListView(self.layoutWidget)
        self.SliceView.setObjectName(_fromUtf8("SliceView"))
        self.gridLayout.addWidget(self.SliceView, 1, 2, 1, 1)
        self.ParamsView = QtGui.QListView(self.layoutWidget)
        self.ParamsView.setObjectName(_fromUtf8("ParamsView"))
        self.gridLayout.addWidget(self.ParamsView, 1, 3, 1, 1)
        self.LabelingView = QtGui.QListView(self.layoutWidget)
        self.LabelingView.setObjectName(_fromUtf8("LabelingView"))
        self.gridLayout.addWidget(self.LabelingView, 1, 4, 1, 1)

        self.retranslateUi(InputSelectionDialog)
        QtCore.QMetaObject.connectSlotsByName(InputSelectionDialog)

    def retranslateUi(self, InputSelectionDialog):
        InputSelectionDialog.setWindowTitle(_translate("InputSelectionDialog", "Dialog", None))
        self.label.setText(_translate("InputSelectionDialog", "Username", None))
        self.label_2.setText(_translate("InputSelectionDialog", "Stack", None))
        self.label_3.setText(_translate("InputSelectionDialog", "Resolution", None))
        self.label_4.setText(_translate("InputSelectionDialog", "Slice Number", None))
        self.label_5.setText(_translate("InputSelectionDialog", "ParameterSet", None))
        self.label_6.setText(_translate("InputSelectionDialog", "Labeling", None))

