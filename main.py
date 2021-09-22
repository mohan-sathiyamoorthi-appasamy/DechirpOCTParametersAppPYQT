# ------------------------------------------------------
# ---------------------- main.py -----------------------
# ------------------------------------------------------
import pyqt5ac
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from scipy.fft import fft, ifft
import cmath
from scipy.signal import hilbert
from scipy import interpolate
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from scipy.signal import chirp, find_peaks, peak_widths
import numpy as np
import random

pyqt5ac.main(rccOptions='', uicOptions='--from-imports', force=False, initPackage=False, config='',
             ioPaths=[['*.ui', '%%FILENAME%%_ui.py'], ['*.qrc', '%%FILENAME%%_rc.py']])

from mainwindow_ui import Ui_MainWindow

class MatplotlibWidget(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        loadUi("mainwindow.ui", self)

        self.pushButton_LoadInterference.clicked.connect(self.pushButton_LoadInterference_clicked)
        self.pushButton_MovingArmData.clicked.connect(self.pushButton_MovingArmData_clicked)
        self.pushButton_ReferenceArm.clicked.connect(self.pushButton_ReferenceArm_clicked)
        self.pushButton_Generate_Dechirp.clicked.connect(self.pushButton_Generate_Dechirp_clicked)
        self.pushButton_OCT_Parameters.clicked.connect(self.pushButton_OCT_Parameters_clicked)

    def pushButton_LoadInterference_clicked(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter(app.tr("Raw Files (*.raw)"))
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_():
            self.fileNames_Interference = dialog.selectedFiles()

    def pushButton_MovingArmData_clicked(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter(app.tr("Raw Files (*.raw)"))
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_():
            self.fileNames_MovingArm = dialog.selectedFiles()

    def pushButton_ReferenceArm_clicked(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter(app.tr("Raw Files (*.raw)"))
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_():
            self.fileNames_ReferenceArm = dialog.selectedFiles()


    def pushButton_Generate_Dechirp_clicked(self):
            allNormPhase = []
            for itr in self.fileNames_Interference:
                Reference = np.fromfile(itr, dtype=np.int16)
                dataSize = (1000, 2048)
                Reference = Reference.reshape(dataSize)
                singleFringe = Reference[0, :]

                # Fourier Transform
                PSF = fft(singleFringe)
                cropSpectra = PSF[5:-1]
                cropSpectra = abs(cropSpectra[1:int(len(cropSpectra) / 2)])

                peaks, _ = find_peaks(cropSpectra, height=0)
                width = np.diff(peaks).max()

                peakLoc = np.where(cropSpectra == np.amax(cropSpectra))

                startPeakLoc = int(peakLoc[0] - (width+5))
                endPeakLoc = int(peakLoc[0] + (width+5))

                filt_fn = np.zeros(PSF.shape)
                filt_fn[startPeakLoc:endPeakLoc, ] = 1

                # Filter the single Peak
                singlePeak = PSF * filt_fn

                # IFFT
                filteredSignal = ifft(singlePeak)

                # Find Phase
                phaseOCT = np.unwrap(np.angle(hilbert((filteredSignal.imag))))

                # Phase normalization
                normPhase = phaseOCT / max(phaseOCT)

                # Sampling Points
                samplingPoints = 2048
                allNormPhase.append(normPhase * samplingPoints)

            meanPhase = sum(allNormPhase) / len(allNormPhase)


            # Rescaling of Average Phase
            avgPhaseRescale = np.zeros([Reference.shape[1]])

            for i in range(Reference.shape[1]):
                avgPhaseRescale[i] = min(meanPhase) + ((max(meanPhase) - min(meanPhase)) / Reference.shape[1]) * i

            nCameraPixels = 2048
            x = np.arange(nCameraPixels)
            f = interpolate.interp1d(meanPhase, x, 'cubic')
            dechirpData = f(avgPhaseRescale)
            dechirpData[dechirpData < 0] = 0

            np.savetxt('Dechirp.txt', [dechirpData], fmt='%0.3f',delimiter='\t')

            self.MplWidget.canvas.axes.clear()
            self.MplWidget.canvas.axes.plot(x, dechirpData)
            self.MplWidget.canvas.draw()
            self.MplWidget.canvas.axes.set_title('Dechirp Data')

    def ReadRawOCTFile(self,filenames):
            bScanRawDataStream = np.fromfile(filenames,dtype = np.int16)
            spectroMeterData = bScanRawDataStream.reshape(1000,2048)
            return spectroMeterData

    def pushButton_OCT_Parameters_clicked(self):
            print('OCT')
            self.MplWidget.canvas.axes.clear()
            nPosition = 7
            for iPosition in range(nPosition):
                interferenceArmData = self.ReadRawOCTFile(self.fileNames_Interference[iPosition])
                MovingArmData = self.ReadRawOCTFile(self.fileNames_MovingArm[iPosition])
                RefArmData = self.ReadRawOCTFile(self.fileNames_ReferenceArm[iPosition])

                # Find Mean of Moving Arm and Reference Arm Data
                avgMovingArmData = MovingArmData.mean(axis=0)
                avgRefArmData = RefArmData.mean(axis=0)

                # Find Spectra of addition of Moving arm and Ref arm Data
                meanSpectra = avgMovingArmData + avgRefArmData

                # Background Subtraction
                fringe = interferenceArmData[0] - meanSpectra

                # Resampled Dechirp Data
                dechirpData = np.loadtxt('Dechirp.txt')

                # Interpolation of fringe with resampled dechirp Data
                x = np.arange(2048)
                vq = interpolate.interp1d(x, fringe, 'cubic')
                resampledFringe = vq(dechirpData)

                # FFT of Windowed Data & crop it
                nCameraPixels = 2048
                hannWindow = np.hanning(nCameraPixels)
                filtData = resampledFringe * hannWindow

                PSFData = fft(filtData)
                calibratedPSF = abs(PSFData)

                # FFT of uncalibrated Fringe
                unCalibratedPSF = abs(fft(fringe))
                print('Hi')
                # Display Data For Sensitivity Roll-off

                self.MplWidget.canvas.axes.plot(x[5:1024],  20 * (np.log10(calibratedPSF[5:1024])))
                self.MplWidget.canvas.draw()
                self.MplWidget.canvas.axes.set_title('Sensitivity Roll-off')

app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()
