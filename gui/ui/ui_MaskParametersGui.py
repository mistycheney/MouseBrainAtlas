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
        MaskParametersGui.resize(446, 300)
        self.gridLayout_2 = QtGui.QGridLayout(MaskParametersGui)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.spinBox_minSize = QtGui.QSpinBox(MaskParametersGui)
        self.spinBox_minSize.setMaximum(10000)
        self.spinBox_minSize.setObjectName(_fromUtf8("spinBox_minSize"))
        self.gridLayout.addWidget(self.spinBox_minSize, 3, 2, 1, 1)
        self.label_7 = QtGui.QLabel(MaskParametersGui)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.gridLayout.addWidget(self.label_7, 0, 3, 1, 1)
        self.label_6 = QtGui.QLabel(MaskParametersGui)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout.addWidget(self.label_6, 0, 2, 1, 1)
        self.spinBox_trainDistPercentile_nissl = QtGui.QSpinBox(MaskParametersGui)
        self.spinBox_trainDistPercentile_nissl.setObjectName(_fromUtf8("spinBox_trainDistPercentile_nissl"))
        self.gridLayout.addWidget(self.spinBox_trainDistPercentile_nissl, 1, 3, 1, 1)
        self.spinBox_trainDistPercentile_fluoro = QtGui.QSpinBox(MaskParametersGui)
        self.spinBox_trainDistPercentile_fluoro.setObjectName(_fromUtf8("spinBox_trainDistPercentile_fluoro"))
        self.gridLayout.addWidget(self.spinBox_trainDistPercentile_fluoro, 1, 2, 1, 1)
        self.spinBox_chi2Threshold_nissl = QtGui.QDoubleSpinBox(MaskParametersGui)
        self.spinBox_chi2Threshold_nissl.setObjectName(_fromUtf8("spinBox_chi2Threshold_nissl"))
        self.gridLayout.addWidget(self.spinBox_chi2Threshold_nissl, 2, 3, 1, 1)
        self.label_2 = QtGui.QLabel(MaskParametersGui)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 2)
        self.label = QtGui.QLabel(MaskParametersGui)
        self.label.setWordWrap(True)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 1, 0, 1, 2)
        self.spinBox_chi2Threshold_fluoro = QtGui.QDoubleSpinBox(MaskParametersGui)
        self.spinBox_chi2Threshold_fluoro.setObjectName(_fromUtf8("spinBox_chi2Threshold_fluoro"))
        self.gridLayout.addWidget(self.spinBox_chi2Threshold_fluoro, 2, 2, 1, 1)
        self.label_3 = QtGui.QLabel(MaskParametersGui)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 2)
        self.button_confirmMaskParameters = QtGui.QPushButton(MaskParametersGui)
        self.button_confirmMaskParameters.setObjectName(_fromUtf8("button_confirmMaskParameters"))
        self.gridLayout_2.addWidget(self.button_confirmMaskParameters, 1, 0, 1, 1)
        self.button_closeMaskParameters = QtGui.QPushButton(MaskParametersGui)
        self.button_closeMaskParameters.setObjectName(_fromUtf8("button_closeMaskParameters"))
        self.gridLayout_2.addWidget(self.button_closeMaskParameters, 1, 1, 1, 1)

        self.retranslateUi(MaskParametersGui)
        QtCore.QMetaObject.connectSlotsByName(MaskParametersGui)

    def retranslateUi(self, MaskParametersGui):
        MaskParametersGui.setWindowTitle(_translate("MaskParametersGui", "Mask Detection Parameters", None))
        self.label_7.setText(_translate("MaskParametersGui", "Nissl", None))
        self.label_6.setText(_translate("MaskParametersGui", "Fluorescent", None))
        self.label_2.setText(_translate("MaskParametersGui", "Threshold on Border Distance to Determine Foreground", None))
        self.label.setText(_translate("MaskParametersGui", "Use Which Percentile of Border Texture Distances", None))
        self.label_3.setText(_translate("MaskParametersGui", "Minimum Size", None))
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

