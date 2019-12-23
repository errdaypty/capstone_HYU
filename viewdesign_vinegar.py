import sys
import cv2
import os
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, QDateTime, Qt, QTimer, QDir, QUrl, pyqtSlot, QThread, pyqtSignal, QDate
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc


# function called by trackbar, sets the next frame to be read

def nothing(x):
    pass


# uic = ui 파일을 py 파일로 변환시켜주는 모듈
form_class = uic.loadUiType("viewdesign.ui")[0]


# 영상 송출 관련은 다 Thread?
# Thread는 알바를 고용하는 것과 같다. 업무 분담에 V ㅔ 리 효율적
# Thread는 지금까지 사용했던 PyQt의 기본 위젯과 달리 사용자가 정의한 클래스이므로 이벤트 역시 사용자가 직접 정의할 수 있다.
# run 메소드는 절대 이름을 바꾸면 안된다.
# Thread 클래스는 정의만 되었기 때문에 실제로 동작하지는 않습니다. 동작하려면 Thread 클래스의 인스턴스를 생성한 후 start() 메서드를 호출해줘야 합니다.
# 여러분이 start() 메서드를 호출하면 Thread 클래스 내부에 정의된 run() 메소드가 호출됩니다.


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    changeSlider = pyqtSignal(int, int)
    moveSlider = pyqtSignal(int)
    presentFrame = pyqtSignal(int)
    giveContour = pyqtSignal(list)
    temporaryReturn = pyqtSignal(QImage)
    storeAllFrames = pyqtSignal(QImage)
    nowFrameToYou = pyqtSignal(int)
    showMessageBox = pyqtSignal()

    def __init__(self, receive=0, percent=0):
        super().__init__()
        self.receive = receive
        self.stop = 0
        self.nowframe = 0
        self.percent = percent

    def receivegotoframe(self, pp):
        self.receive = pp

    def receivefilename(self, filename):
        self.filename = filename

    def receivepercent(self, percent):
        self.percent = percent

    def run(self):

        global h, w
        cap = cv2.VideoCapture(self.filename)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.receive)
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_WIDTH * self.percent/100)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_HEIGHT * self.percent/100)
            beginning_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            nr_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # frame_list = list(range(nr_of_frames))
            self.tfc = nr_of_frames
            self.frameList = list(range(beginning_frame, nr_of_frames))

            # get total number of frames cap.get : 속성반환
            # 트랙바 도대체 왜 안생김?? run 안이나 play() 안에 넣어도 안생김
            # cv2.createTrackbar("Video","cap",0,nr_of_frames, getFrame)
            # self.videoSlider.setValue(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            # for i in range(nr_of_frames):

            for i in self.frameList:
                nowframe = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.nowFrameToYou.emit(nowframe)
                ret, frame = cap.read()
                if ret:
                    # 이건 resolution 조절인듯...
                    width = int(frame.shape[1])
                    height = int(frame.shape[0])

                    if self.percent != 0:
                        cutframe = frame[int(height * self.percent / 2 / 100 / 1.85):-int(
                            height * self.percent / 2 / 100 / 1.85),
                                   int(width * self.percent / 2 / 100):int(-width * self.percent / 2 / 100)]
                    else:
                        cutframe = frame

                    # dim = (width,height)
                    # cutframe = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                    # frame[y,x]
                    # frame[118:,300:-85] for 'testfly3.avi'
                    # cutframe = frame[50:,240:-150]
                    grayimg = cv2.cvtColor(cutframe, cv2.COLOR_BGR2GRAY)
                    opening = cv2.adaptiveThreshold(grayimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                    15, 5)
                    median_blur = cv2.medianBlur(opening, 3)
                    # Threshold를 했으니, single channel image가 된 것 아닌가?
                    # threshold 이전에 grayscale로 바꿔주니 됨. threshold src가 애초에 grayscale이 아니였는듯
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

                    eroded_image = cv2.erode(median_blur, kernel, 1)
                    # dilated_image = cv2.dilate(median_blur,kernel,1)
                    opened_image = cv2.dilate(eroded_image, kernel, 1)
                    cnts, _ = cv2.findContours(opened_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    # cv2.CHAIN_APPROX_SIMPLE - compresses horizontal, vertical, and diagonal segments and leaves only their end points.
                    # For example, an up-right rectangular contour is encoded with 4 points.
                    # cv2.drawContours(grayimg, cnts, -1, (255, 0, 0), 2)
                    # cv2.polylines(grayimg, cnts,isClosed=True,color=(255,0,0),thickness = 1)

                    rgbImage = cv2.cvtColor(grayimg, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                    p = convertToQtFormat.scaled(600, 350, Qt.KeepAspectRatio)
                    # p = convertToQtFormat.scaled(100, 100, Qt.KeepAspectRatio)

                    # changePixmap 이란 QImage에 대한 사용자 정의 시그널을 발생(emit)시킵니다. p 전송.
                    if self.stop == True:
                        break
                    self.changePixmap.emit(p)
                    k = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.nowframe = k
                    self.changeSlider.emit(k, nr_of_frames)
                    self.presentFrame.emit(k)
                    self.giveContour.emit(cnts)
                    self.storeAllFrames.emit(p)
        else:
            self.showMessageBox.emit()


class MyWindow(QMainWindow, form_class):
    send_File = pyqtSignal(str)
    go_To_Frame = pyqtSignal(int)
    S_control_Frame_Ratio = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)

        self.iscamconnected_L.setEnabled(False)

        # self.play_PB.setEnabled(False)
        self.play_PB.toggled.connect(self.play_Video)
        self.playcount = 0
        self.state = 0

        self.actionOpen.setShortcut('Ctrl+O')
        self.actionOpen.triggered.connect(self.openFile)

        self.record_PB.clicked.connect(self.record)

        self.videoSlider.setMinimum(0)
        self.videoSlider.sliderMoved.connect(self.temporary_Slider_Move)
        self.videoSlider.sliderReleased.connect(self.go_To)

        now = QDate.currentDate()
        self.date_L.setText(now.toString(Qt.ISODate))

        self.ratioSlider.setMinimum(0)
        self.ratioSlider.sliderMoved.connect(self.control_Frame_Ratio)

        self.enablepcs_PB.setEnabled(False)
        self.enablepcs_PB.toggled.connect(self.toggle_ImageProcessStart)

        # list for storing contours
        self.a = []
        # list for storing frames
        self.b = []

        self.coordination_array = []
        self.initiation = []
        # ============== From Main to Worker Thread ==============
        # filename 을 반환하기 위한 thread 선언. 근데 그냥 포기할까
        # 송출한 것을 Thread의 메쏘드로 연결
        self.th = Thread()
        self.send_File.connect(self.th.receivefilename)
        self.go_To_Frame.connect(self.th.receivegotoframe)
        self.S_control_Frame_Ratio.connect(self.th.receivepercent)
        # ============== From Main to Worker Thread ===============

    def setImage(self, image):
        if image in self.b:
            self.video_L.setPixmap(QPixmap.fromImage(self.b[int(self.videoSlider.value()) - 1]))
        else:
            self.video_L.setPixmap(QPixmap.fromImage(image))

    def openFile(self):
        fileName = QFileDialog.getOpenFileName(self, "Open Movie", QDir.homePath())

        if fileName != '':
            videoName = fileName[0]
            videoName_toshow = os.path.basename(str(fileName[0]))
            self.filename_L.setText(videoName_toshow)
            self.send_File.emit(videoName)

        self.play_PB.setEnabled(True)

    def messageBoxInThread(self, ):
        QMessageBox.about(self, "Error", "비디오파일이 아닙니다.")

    def play_Video(self, state):
        self.enablepcs_PB.setEnabled(True)
        self.play_PB.setText({True: "STOP", False: "PLAY"}[state])
        self.play_PB.setIcon(QIcon({True: '&stop.png', False: "&play_button.png"}[state]))

        if state:
            if self.playcount >= 1:
                if self.th.nowframe != 0:
                    self.th.__init__(self.th.nowframe, self.ratioSlider.value())
                    self.th.giveContour.connect(self.scatter_ImageProcess)
                    self.th.nowFrameToYou.connect(self.windrose_ImageProcess)

            # ------------from Worker to Main ------------
            # changePixmap(사용자 정의 시그널)이 발생하면 setImage 메서드 호출

            self.th.changePixmap.connect(self.setImage)
            self.th.storeAllFrames.connect(self.storeFrames)
            self.th.changeSlider.connect(self.rt_value_changed)
            self.th.presentFrame.connect(self.progressBar.setValue)
            self.th.temporaryReturn.connect(self.setImage)
            self.th.giveContour.connect(self.storeContours)
            self.th.showMessageBox.connect(self.messageBoxInThread)
            self.th.start()
            self.playcount += 1

            # ------------form Worker to Main -----------

        else:
            self.th.stop = 1

    def temporary_Slider_Move(self):
        self.setImage(self.b[int(self.videoSlider.value()) - 1])

    def go_To(self):

        target_frame = int(self.videoSlider.value())
        self.th.nowframe = target_frame
        self.th.percent = self.ratioSlider.value()
        self.go_To_Frame.emit(target_frame)
        self.th.giveContour.connect(self.scatter_ImageProcess)
        self.th.nowFrameToYou.connect(self.windrose_ImageProcess)

    def rt_value_changed(self, k, nr_of_frames):
        self.videoSlider.setMaximum(nr_of_frames)
        self.videoSlider.setValue(k)
        self.progressBar.setRange(0, nr_of_frames)

        if k == nr_of_frames:
            self.play_PB.setChecked(False)
            self.play_PB.setText('REPLAY')
            self.playcount = 0
            self.th.stop = 0
            self.th.__init__()
            # print('playcount = ',self.playcount)
            # self.a = []
            # self.b = []
            # 레이쇼 슬라이더도 초기화

    def record(self):
        QMessageBox.about(self, "warning", "카메라가 연결돼 있지 않습니다")

    def control_Frame_Ratio(self):
        self.S_control_Frame_Ratio.emit(int(self.ratioSlider.value()))

    def storeContours(self, cnts):
        self.a.append(cnts)

    def storeFrames(self, frames):
        self.b.append(frames)

    def toggle_ImageProcessStart(self, state):
        self.enablepcs_PB.setText({False: "IMAGE PROCESS", True: "OFF"}[state])

        if state:
            self.imageProcessStart()

        else:
            self.lay.deleteLater()
            self.lay2.deleteLater()
            self.lay.removeWidget(self.canvas)
            self.lay2.removeWidget(self.canvas2)
            self.canvas.setParent(None)
            self.canvas2.setParent(None)

    def imageProcessStart(self):

        self.fig, self.ax = plt.subplots()

        # 도대체 왜 subplots(0)일 때랑 타입이 다른거지???
        self.canvas = FigureCanvas(self.fig)

        ###### Scatter plot on the GUI

        self.lay = QGridLayout()
        self.lay.addWidget(self.canvas)
        self.scatter_W.setLayout(self.lay)
        self.canvas.draw()

        ###### Scatter plot on the GUI
        ln, = self.ax.plot([], [], 'ro', markersize=2)
        self.th.giveContour.connect(self.scatter_ImageProcess)

        def init():
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(0, h)
            self.ax.set_ylim(self.ax.get_ylim()[::-1])
            self.ax.set_title('Scatter Plot')
            return ln,

        def update(frame):
            # plot([[1,2,3,4,,,,,]],[[1,5,2,3,4,3,,,4,]])
            ln.set_data(self.xdata, self.ydata)
            return ln,

        self.ani = animation.FuncAnimation(self.fig, update, frames=self.th.tfc, init_func=init, blit=True)
        self.canvas.show()
        ###### Scatter plot on the GUI

        #################### Windrose Plot on the GUI ###############################

        self.fig2, self.ax2 = plt.subplots(1)

        self.canvas2 = FigureCanvas(self.fig2)
        self.lay2 = QGridLayout()
        self.lay2.addWidget(self.canvas2)
        self.windrose_W.setLayout(self.lay2)
        self.canvas2.draw()

        origin = [0], [0]
        Q = self.ax2.quiver(*origin, [], [], color='m', scale=1000)
        self.th.nowFrameToYou.connect(self.windrose_ImageProcess)

        def init2():
            # ax2.set_xlim(0,w)
            # ax2.set_ylim(0,h)
            # ax2.set_ylim(ax.get_ylim()[::-1])
            self.ax2.set_title('Windrose Plot')
            return Q,

        def update2(frame):
            Q.set_UVC(self.scaled_chosen_xdata2, self.scaled_chosen_ydata2)
            return Q,

        """
            if (cX2 >= w/2 and cX2 <=w) and (cY2>=0 and cY2<=h/2):
                 if (cX2>=w/2 and cX2<=w/2+h/2) and (cY2>=0 and cY2 <= -cX2 + h/2+w/2):
                     radii.append(len(cX2)+len(cY2))
                 else:
                     radii.append(len(cX2)+len(cY2))

            if (cX2 >= 0 and cX2 <= w/2) and (cY2 >= 0 and cY2<=h / 2):
                if (cX2 >= w/2 - h/2 and cX2 <= w / 2 ) and (cY2 >= 0 and cY2 <= cX2 -w/2 + h / 2):
                    radii.append(len(cX2) + len(cY2))
                else:
                    radii.append(len(cX2)+len(cY2))

            if (cX2 >= 0 and cX2 <= w/2) and (cY2 >= h/2 and cY2 <=h):
                if (cX2 >= w/2 - h/2 and cX2 <= w/2 ) and (cY2 >= h/2 and cY2 >= -cX2 + h/2 + w/2):
                    radii.append(len(cX2) + len(cY2))
                else:
                    radii.append(len(cX2)+len(cY2))

            if (cX2 >= w / 2 and cX2 <= w) and (cY2 >= h/2 and cY2 <= h):
                if (cX2 >= w / 2 and cX2 <= w / 2 + h / 2) and (cY2 >= h/2 and cY2 >= cX2 - w/2 + h/2):
                    radii.append(len(cX2) + len(cY2))
                else:
                    radii.append(len(cX2)+len(cY2))

            # plot([[1,2,3,4,,,,,]],[[1,5,2,3,4,3,,,4,]])
            return radii
        """

        self.ani2 = animation.FuncAnimation(self.fig2, update2, frames=self.th.tfc, init_func=init2, blit=True)
        self.canvas2.show()

        # th.giveContour가 for 안에 있으므로, scatter_ImageProcess도 반복반복
        #    QMessageBox.about(self, "warning", "비디오가 없습니다")

    def scatter_ImageProcess(self, cnts):
        self.xdata = []
        self.ydata = []
        for a, b in enumerate(cnts):

            M = cv2.moments(b)
            # (dcX, dcY), (width, height), _ = cv2.fitEllipse(b)
            parameter = cv2.arcLength(b, True)
            # print('width',width)
            # print('height',height)
            # contourArea < 150 하니까 거의 다 보임. 원래는 < 100
            # 초파리 중간 크기 WIdth = 24.07 px (장축)
            # <150. <25, <25 하니 상당히 정확함
            if 20 < cv2.contourArea(b) < 150 and M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                self.xdata.append(cX)
                self.ydata.append(cY)

    def windrose_ImageProcess(self, nowframe):

            if nowframe >= 1:

                self.xdata2 = []
                self.ydata2 = []
                self.coordinate_list = []

                # self.initiation - processing 버튼을 누른 이후의 프레임들이 저장되는 리스트
                self.initiation.append(nowframe)

                for a, b in enumerate(self.a[nowframe - 1]):
                    M = cv2.moments(b)
                    # (dcX, dcY), (width, height), _ = cv2.fitEllipse(b)
                    parameter = cv2.arcLength(b, True)
                    if 20 < cv2.contourArea(b) < 150 and M["m00"] != 0:
                        cX2 = int(M["m10"] / M["m00"])
                        cY2 = int(M["m01"] / M["m00"])
                        self.xdata2.append(cX2)
                        self.ydata2.append(cY2)

                self.coordination_list = np.array([self.xdata2, self.ydata2])

                self.coordination_array.append(self.coordination_list)

                self.chosen_xdata2 = []
                self.chosen_ydata2 = []

                if len(self.coordination_array) >= 2:
                    # 얘네 type은 ndarray
                    lastCx = self.coordination_array[self.initiation.index(nowframe) - 1][0, :]
                    lastCy = self.coordination_array[self.initiation.index(nowframe) - 1][1, :]
                    nowCx = self.coordination_array[self.initiation.index(nowframe)][0, :]
                    nowCy = self.coordination_array[self.initiation.index(nowframe)][1, :]



                    # 각 좌표의 ndarray를 반복해서 만든 matrix
                    # 열 형태로 반복
                    lastxM = np.repeat(lastCx, len(nowCx)).reshape((len(lastCx), len(nowCx)))
                    lastyM = np.repeat(lastCy, len(nowCy)).reshape((len(lastCy), len(nowCy)))
                    # 행 형태로 반복
                    nowxM = np.tile(nowCx, len(lastCx)).reshape((len(lastCx), len(nowCx)))
                    nowyM = np.tile(nowCy, len(lastCy)).reshape((len(lastCy), len(nowCy)))

                    distanceM = np.sqrt((lastxM - nowxM) ** 2 + (lastyM - nowyM) ** 2)

                    dcompareM = np.abs(nowxM - lastxM) - np.abs(nowyM - lastyM)
                    filteredM = np.where(dcompareM>=0, 1, 0)
                    print('filterd',filteredM)
                    minIndexArray = np.argmin(distanceM, axis=1)
                    minArray = np.min(distanceM, axis=1)

                    flat_filteredM = filteredM.ravel()
                    asfasd = np.argmin(distanceM)

                    minDict = dict(zip(minIndexArray, minArray))
                    minFilteredDict = dict(zip(minIndexArray,filteredM.ravel()))


                    self.chosen_xdata2.append([nowCx[numbering] for numbering, value in minDict.items() if
                                               value > 0 ])
                    self.chosen_ydata2.append([nowCy[numbering] for numbering, value in minDict.items() if
                                               value > 0 ])

                self.scaled_chosen_xdata2 = np.array(self.chosen_xdata2) - w / 2
                self.scaled_chosen_ydata2 = np.array(self.chosen_ydata2) - h / 2


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()