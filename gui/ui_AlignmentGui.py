# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AlignmentGui.ui'
#
# Created: Wed Sep 14 01:46:02 2016
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

class Ui_AlignmentGui(object):
    def setupUi(self, AlignmentGui):
        AlignmentGui.setObjectName(_fromUtf8("AlignmentGui"))
        AlignmentGui.resize(1390, 972)
        self.gridLayout = QtGui.QGridLayout(AlignmentGui)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.label_2 = QtGui.QLabel(AlignmentGui)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout_2.addWidget(self.label_2)
        self.prev_gview = QtGui.QGraphicsView(AlignmentGui)
        self.prev_gview.setObjectName(_fromUtf8("prev_gview"))
        self.verticalLayout_2.addWidget(self.prev_gview)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_previous_filename = QtGui.QLabel(AlignmentGui)
        self.label_previous_filename.setObjectName(_fromUtf8("label_previous_filename"))
        self.horizontalLayout_3.addWidget(self.label_previous_filename)
        self.label_previous_index = QtGui.QLabel(AlignmentGui)
        self.label_previous_index.setObjectName(_fromUtf8("label_previous_index"))
        self.horizontalLayout_3.addWidget(self.label_previous_index)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.label_3 = QtGui.QLabel(AlignmentGui)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.verticalLayout_3.addWidget(self.label_3)
        self.aligned_gview = QtGui.QGraphicsView(AlignmentGui)
        self.aligned_gview.setObjectName(_fromUtf8("aligned_gview"))
        self.verticalLayout_3.addWidget(self.aligned_gview)
        self.gridLayout.addLayout(self.verticalLayout_3, 0, 3, 1, 1)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label = QtGui.QLabel(AlignmentGui)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout.addWidget(self.label)
        self.curr_gview = QtGui.QGraphicsView(AlignmentGui)
        self.curr_gview.setObjectName(_fromUtf8("curr_gview"))
        self.verticalLayout.addWidget(self.curr_gview)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_current_filename = QtGui.QLabel(AlignmentGui)
        self.label_current_filename.setObjectName(_fromUtf8("label_current_filename"))
        self.horizontalLayout_2.addWidget(self.label_current_filename)
        self.label_current_index = QtGui.QLabel(AlignmentGui)
        self.label_current_index.setObjectName(_fromUtf8("label_current_index"))
        self.horizontalLayout_2.addWidget(self.label_current_index)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout.addLayout(self.verticalLayout, 0, 2, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_4 = QtGui.QLabel(AlignmentGui)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout.addWidget(self.label_4)
        self.comboBox_parameters = QtGui.QComboBox(AlignmentGui)
        self.comboBox_parameters.setObjectName(_fromUtf8("comboBox_parameters"))
        self.comboBox_parameters.addItem(_fromUtf8(""))
        self.comboBox_parameters.addItem(_fromUtf8(""))
        self.comboBox_parameters.addItem(_fromUtf8(""))
        self.comboBox_parameters.addItem(_fromUtf8(""))
        self.comboBox_parameters.addItem(_fromUtf8(""))
        self.comboBox_parameters.addItem(_fromUtf8(""))
        self.comboBox_parameters.addItem(_fromUtf8(""))
        self.horizontalLayout.addWidget(self.comboBox_parameters)
        self.button_align = QtGui.QPushButton(AlignmentGui)
        self.button_align.setObjectName(_fromUtf8("button_align"))
        self.horizontalLayout.addWidget(self.button_align)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 2, 1, 1)
        self.button_anchor = QtGui.QPushButton(AlignmentGui)
        self.button_anchor.setObjectName(_fromUtf8("button_anchor"))
        self.gridLayout.addWidget(self.button_anchor, 1, 1, 1, 1)
        self.button_upload_transform = QtGui.QPushButton(AlignmentGui)
        self.button_upload_transform.setObjectName(_fromUtf8("button_upload_transform"))
        self.gridLayout.addWidget(self.button_upload_transform, 1, 3, 1, 1)

        self.retranslateUi(AlignmentGui)
        QtCore.QMetaObject.connectSlotsByName(AlignmentGui)

    def retranslateUi(self, AlignmentGui):
        AlignmentGui.setWindowTitle(_translate("AlignmentGui", "Dialog", None))
        self.label_2.setText(_translate("AlignmentGui", "Previous", None))
        self.label_previous_filename.setText(_translate("AlignmentGui", "TextLabel", None))
        self.label_previous_index.setText(_translate("AlignmentGui", "TextLabel", None))
        self.label_3.setText(_translate("AlignmentGui", "Current aligned to Previous", None))
        self.label.setText(_translate("AlignmentGui", "Current", None))
        self.label_current_filename.setText(_translate("AlignmentGui", "TextLabel", None))
        self.label_current_index.setText(_translate("AlignmentGui", "TextLabel", None))
        self.label_4.setText(_translate("AlignmentGui", "Elastix parameters", None))
        self.comboBox_parameters.setItemText(0, _translate("AlignmentGui", "Rigid", None))
        self.comboBox_parameters.setItemText(1, _translate("AlignmentGui", "Rigid_MutualInformation", None))
        self.comboBox_parameters.setItemText(2, _translate("AlignmentGui", "Rigid_RequiredRatioOfValidSamples", None))
        self.comboBox_parameters.setItemText(3, _translate("AlignmentGui", "Rigid_noNumberOfSpatialSamples", None))
        self.comboBox_parameters.setItemText(4, _translate("AlignmentGui", "Affine", None))
        self.comboBox_parameters.setItemText(5, _translate("AlignmentGui", "BSpline", None))
        self.comboBox_parameters.setItemText(6, _translate("AlignmentGui", "Translation", None))
        self.button_align.setText(_translate("AlignmentGui", "align", None))
        self.button_anchor.setText(_translate("AlignmentGui", "Add anchor pair", None))
        self.button_upload_transform.setText(_translate("AlignmentGui", "Upload transform to server", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    AlignmentGui = QtGui.QDialog()
    ui = Ui_AlignmentGui()
    ui.setupUi(AlignmentGui)
    AlignmentGui.show()
    sys.exit(app.exec_())

