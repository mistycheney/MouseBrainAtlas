# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MaskParametersGui.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
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

class Ui_MaskParametersGui(object):
    def setupUi(self, MaskParametersGui):
        MaskParametersGui.setObjectName(_fromUtf8("MaskParametersGui"))
        MaskParametersGui.resize(400, 300)
        self.gridLayout_2 = QtGui.QGridLayout(MaskParametersGui)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label = QtGui.QLabel(MaskParametersGui)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.spinBox_trainDistPercentile = QtGui.QSpinBox(MaskParametersGui)
        self.spinBox_trainDistPercentile.setObjectName(_fromUtf8("spinBox_trainDistPercentile"))
        self.gridLayout.addWidget(self.spinBox_trainDistPercentile, 0, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.button_confirmMaskParameters = QtGui.QPushButton(MaskParametersGui)
        self.button_confirmMaskParameters.setObjectName(_fromUtf8("button_confirmMaskParameters"))
        self.gridLayout_2.addWidget(self.button_confirmMaskParameters, 1, 1, 1, 1)
        self.button_closeMaskParameters = QtGui.QPushButton(MaskParametersGui)
        self.button_closeMaskParameters.setObjectName(_fromUtf8("button_closeMaskParameters"))
        self.gridLayout_2.addWidget(self.button_closeMaskParameters, 1, 2, 1, 1)

        self.retranslateUi(MaskParametersGui)
        QtCore.QMetaObject.connectSlotsByName(MaskParametersGui)

    def retranslateUi(self, MaskParametersGui):
        MaskParametersGui.setWindowTitle(_translate("MaskParametersGui", "Dialog", None))
        self.label.setText(_translate("MaskParametersGui", "Train distance percentile", None))
        self.button_confirmMaskParameters.setText(_translate("MaskParametersGui", "Confirm", None))
        self.button_closeMaskParameters.setText(_translate("MaskParametersGui", "Quit", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MaskParametersGui = QtGui.QDialog()
    ui = Ui_MaskParametersGui()
    ui.setupUi(MaskParametersGui)
    MaskParametersGui.show()
    sys.exit(app.exec_())

