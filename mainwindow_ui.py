# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Mohan\PycharmProjects\TestPlotWindow\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(526, 645)
        MainWindow.setMinimumSize(QtCore.QSize(500, 500))
        MainWindow.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(30, 470, 181, 80))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.pushButton_Generate_Dechirp = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_Generate_Dechirp.setMinimumSize(QtCore.QSize(120, 40))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.pushButton_Generate_Dechirp.setFont(font)
        self.pushButton_Generate_Dechirp.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 75 8pt \"MS Shell Dlg 2\";\n"
"background-color: rgb(170, 170, 127);")
        self.pushButton_Generate_Dechirp.setObjectName("pushButton_Generate_Dechirp")
        self.horizontalLayout.addWidget(self.pushButton_Generate_Dechirp)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.MplWidget = MplWidget(self.centralwidget)
        self.MplWidget.setGeometry(QtCore.QRect(30, 130, 480, 320))
        self.MplWidget.setMinimumSize(QtCore.QSize(480, 320))
        self.MplWidget.setObjectName("MplWidget")
        self.pushButton_LoadInterference = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_LoadInterference.setGeometry(QtCore.QRect(30, 60, 141, 51))
        self.pushButton_LoadInterference.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 75 8pt \"MS Shell Dlg 2\";\n"
"background-color: rgb(170, 170, 127);")
        self.pushButton_LoadInterference.setObjectName("pushButton_LoadInterference")
        self.pushButton_MovingArmData = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_MovingArmData.setGeometry(QtCore.QRect(200, 60, 141, 51))
        self.pushButton_MovingArmData.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 75 8pt \"MS Shell Dlg 2\";\n"
"background-color: rgb(170, 170, 127);")
        self.pushButton_MovingArmData.setObjectName("pushButton_MovingArmData")
        self.pushButton_ReferenceArm = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_ReferenceArm.setGeometry(QtCore.QRect(370, 60, 141, 51))
        self.pushButton_ReferenceArm.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 75 8pt \"MS Shell Dlg 2\";\n"
"background-color: rgb(170, 170, 127);")
        self.pushButton_ReferenceArm.setObjectName("pushButton_ReferenceArm")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(300, 470, 181, 80))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.pushButton_OCT_Parameters = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.pushButton_OCT_Parameters.setMinimumSize(QtCore.QSize(120, 40))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.pushButton_OCT_Parameters.setFont(font)
        self.pushButton_OCT_Parameters.setStyleSheet("color: rgb(0, 0, 0);\n"
"font: 75 8pt \"MS Shell Dlg 2\";\n"
"background-color: rgb(170, 170, 127);")
        self.pushButton_OCT_Parameters.setObjectName("pushButton_OCT_Parameters")
        self.horizontalLayout_2.addWidget(self.pushButton_OCT_Parameters)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_Generate_Dechirp.setText(_translate("MainWindow", " Dechirp Generation"))
        self.pushButton_LoadInterference.setText(_translate("MainWindow", "Load Interference Data"))
        self.pushButton_MovingArmData.setText(_translate("MainWindow", "Load Moving Arm Data"))
        self.pushButton_ReferenceArm.setText(_translate("MainWindow", "Load Reference Arm Data"))
        self.pushButton_OCT_Parameters.setText(_translate("MainWindow", "OCT Parameters"))
from mplwidget import MplWidget
