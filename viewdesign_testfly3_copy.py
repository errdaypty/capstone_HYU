import sys
import cv2
import os
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, QDateTime, Qt, QTimer, QDir, QUrl, pyqtSlot, QThread, pyqtSignal, QDate
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import math

from matplotlib import animation, rc
# from matplotlibwidgetFlie import matplotlibwidget

import matplotlib.pyplot as plt


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
    playAgain = pyqtSignal(int)
    giveContour = pyqtSignal(list)
    giveContour2 = pyqtSignal(list)
    temporaryReturn = pyqtSignal(QImage)
    storeAllContours = pyqtSignal(list)
    nowFrameToYou = pyqtSignal(int)

    # frameSize = pyqtSignal(int,int)
    # def getFileName(self,fileName):
    # global cap
    def __init__(self, receive=0):
        super().__init__()
        self.receive = receive
        print('value, self receive', self.receive)
        self.stop = 0
        self.nowframe = 0
        global ss
        ss = self.receive
        self.k = True

    # cap = cv2.VideoCapture('fileName')
    # return cap
    def receivegotoframe(self, pp):
        self.receive = pp

    def returntemporaryframe(self, kkk):
        cap = cv2.VideoCapture('testfly3.avi')
        cap.set(cv2.CAP_PROP_POS_FRAMES, kkk)

    def stoptheloop(self):
        self.k = False

    def run(self):
        global h, w
        cap = cv2.VideoCapture('testfly3.avi')
        cap.set(cv2.CAP_PROP_POS_FRAMES, ss)
        beginning_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        nr_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame_list = list(range(nr_of_frames))
        self.tfc = nr_of_frames
        # get total number of frames cap.get : 속성반환
        # 트랙바 도대체 왜 안생김?? run 안이나 play() 안에 넣어도 안생김
        # cv2.createTrackbar("Video","cap",0,nr_of_frames, getFrame)
        # self.videoSlider.setValue(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        # for i in range(nr_of_frames):
        self.frameList = list(range(beginning_frame, nr_of_frames))
        for i in self.frameList:
            self.nowframe = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print('nowframe', self.nowframe)
            # 왜 presentframe을 windrose에 연결하면 에러발생???
            self.nowFrameToYou.emit(self.nowframe)
            ret, frame = cap.read()
            # frame[y,x]
            # frame[118:,300:-85] for 'testfly3.avi'
            if ret:
                cutframe = frame[118:, 300:-85]
                grayimg = cv2.cvtColor(cutframe, cv2.COLOR_BGR2GRAY)
                # adaptiveTHreshold(_,_,_,...,n) n이 클수록 필터링 효과 크다
                opening = cv2.adaptiveThreshold(grayimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 55,
                                                5)
                median_blur = cv2.medianBlur(opening, 15)
                # Threshold를 했으니, single channel image가 된 것 아닌가?
                # threshold 이전에 grayscale로 바꿔주니 됨. threshold src가 애초에 grayscale이 아니였는듯
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

                eroded_image = cv2.erode(median_blur, kernel, 1)
                # dilated_image = cv2.dilate(median_blur,kernel,1)
                opened_image = cv2.dilate(eroded_image, kernel, 1)
                cnts, _ = cv2.findContours(opened_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # cv2.CHAIN_APPROX_SIMPLE - compresses horizontal, vertical, and diagonal segments and leaves only their end points.
                # For example, an up-right rectangular contour is encoded with 4 points.
                # cv2.drawContours(grayimg, cnts, -1, (255, 0, 0), 2)
                # cv2.polylines(grayimg, cnts,isClosed=True,color=(255,0,0),thickness = 2)
                rgbImage = cv2.cvtColor(grayimg, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                p = convertToQtFormat.scaled(600, 350, Qt.KeepAspectRatio)

                cntt = cnts
                a = cntt

                # changePixmap 이란 QImage에 대한 사용자 정의 시그널을 발생(emit)시킵니다. p 전송.
                if self.stop == True:
                    print('멈춰')
                    break
                    # self.playAgain.emit(0)

                k = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.nowframe = k

                self.changePixmap.emit(p)
                self.changeSlider.emit(k, nr_of_frames)
                self.presentFrame.emit(k)
                self.giveContour.emit(cnts)
                self.giveContour2.emit(cnts)
                self.storeAllContours.emit(cnts)
                # self.moveSlider.emit(pr_of_frames)
                print('k', k)
            else:
                QMessageBox.about(self, "warning", "비디오를 읽어들일 수가 없습니다")


class MyWindow(QMainWindow, form_class):
    send_File = pyqtSignal(str)
    go_To_Frame = pyqtSignal(int)
    video_Pause_Signal = pyqtSignal(bool)
    temporary_Slider_Move_Signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)

        self.play_PB.setCheckable(True)
        self.play_PB.toggled.connect(self.play_Video)
        self.playcount = 0

        self.pause_PB.clicked.connect(self.pause_Video)

        self.actionOpen.setShortcut('Ctrl+O')
        self.actionOpen.triggered.connect(self.openFile)

        self.record_PB.clicked.connect(self.record)
        # self.enablepcs_PB.clicked.connect(self.imageProcess)

        self.videoSlider.setMinimum(0)
        self.videoSlider.sliderMoved.connect(self.temporary_Slider_Move)
        self.videoSlider.sliderReleased.connect(self.go_To)

        now = QDate.currentDate()
        self.date_L.setText(now.toString(Qt.ISODate))
        # self.videoSlider.setValue()

        self.enablepcs_PB.setCheckable(True)
        self.enablepcs_PB.toggled.connect(self.imageProcessStart)

        #### for windrose
        self.a = []
        self.coordination_array = []
        self.initation = []
        # ============== From Main to Worker Thread ==============
        # filename 을 반환하기 위한 thread 선언. 근데 그냥 포기할까
        # 송출한 것을 Thread의 메쏘드로 연결
        self.th = Thread()
        self.send_File.connect(self.th.run)
        self.video_Pause_Signal.connect(self.th.run)
        self.temporary_Slider_Move_Signal.connect(self.th.returntemporaryframe)
        self.go_To_Frame.connect(self.th.receivegotoframe)
        # ============== From Main to Worker Thread ===============

        self.th.playAgain.connect(self.play_Video)

    def setImage(self, image):
        self.video_L.setPixmap(QPixmap.fromImage(image))

    def openFile(self):
        fileName = QFileDialog.getOpenFileName(self, "Open Movie", QDir.homePath())

        # fileName의 데이터 타입은 Tuple이다!!!!
        if fileName != '':
            # 경로에서 파일 아름만 추출하는법...
            video_name = os.path.basename(str(fileName))
            self.filename_L.setText(video_name)
            print(type(video_name))
            print(video_name)
            # 사용자 정의 signal 송출
            # self.send_File.emit(video_name)

        self.play_PB.setEnabled(True)

    def play_Video(self, state):
        # ------------from Worker to Main ------------
        # changePixmap(사용자 정의 시그널)이 발생하면 setImage 메서드 호출
        if state:
            if self.playcount >= 1:
                if self.th.nowframe != 0:
                    self.th.__init__(self.th.nowframe)
                else:
                    self.th.__init__(self.th.receive)

            self.th.changePixmap.connect(self.setImage)
            self.th.changeSlider.connect(self.rt_value_changed)
            self.th.presentFrame.connect(self.progressBar.setValue)  # dasdfasdfasdfasdfasd
            self.th.temporaryReturn.connect(self.setImage)  # asdfasdfasdfasdfasd
            self.th.storeAllContours.connect(self.storeContours)
            self.th.start()
            self.playcount += 1
        else:
            self.th.stop = 1

        # self.th.presentFrame.connect(self.go_To)

    # self.th.moveSlider.connect(self.moveSliderSlot)
    # ------------form Worker to Main -----------
    def pause_Video(self):
        # 왜 thread 종료가 안 되나??
        print('없어질거임 ')
        # 왜 멈춰 안나옴

    def record(self):
        QMessageBox.about(self, "Warning", "파일을 선택하지 않았습니다.")

    def temporary_Slider_Move(self, image):
        temp_frame = int(self.videoSlider.value())
        self.temporary_Slider_Move_Signal.emit(temp_frame)

    def go_To(self):
        target_frame = int(self.videoSlider.value())
        self.th.nowframe = target_frame
        self.go_To_Frame.emit(target_frame)
        self.th.__init__(target_frame)

    def rt_value_changed(self, k, nr_of_frames):
        self.videoSlider.setMaximum(nr_of_frames)
        self.videoSlider.setValue(k)
        self.progressBar.setRange(0, nr_of_frames)

    def record(self):
        QMessageBox.about(self, "warning", "파일을 선택하지 않았습니다")

    def storeContours(self, cnts):
        self.a.append(cnts)

    def toggled_ImageProcessStart(self, state):
        self.connect({True: self.imageProcessStart, False: "OFF"}[state])

    def imageProcessStart(self, state):
        # print(state)
        if state == True:

            self.fig = plt.figure(0)
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvas(self.fig)
            self.lay = QGridLayout()
            self.lay.addWidget(self.canvas)
            self.scatter_W.setLayout(self.lay)
            self.canvas.draw()

            ###### Scatter plot on the GUI
            ln, = plt.plot([], [], 'ro', markersize=2)
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

            self.fig2 = plt.figure(1)
            self.ax2 = self.fig2.add_subplot(111)
            self.canvas2 = FigureCanvas(self.fig2)
            self.lay2 = QGridLayout()
            self.lay2.addWidget(self.canvas2)
            self.windrose_W.setLayout(self.lay2)
            self.canvas2.draw()

            origin = [0], [0]
            Q = self.ax2.quiver(*origin, [], [], color='m', scale=1000)
            # self.th.giveContour2.connect(self.windrose_ImageProcess)
            # 얘 때문에 윈드로즈 반복 수행
            self.th.nowFrameToYou.connect(self.windrose_ImageProcess)
            """
            N = 8
            theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
            width = 2*np.pi / 8
            #colors = plt.cm.viridis(radii2 / 10.)
            #ax2 = plt.subplot(111, projection='polar')
            bar = ax2.bar(theta, [], width=width, bottom=0.0, color='r', alpha=0.5)
            """

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
        else:
            # self.lay.removeItem(self.canvas)
            # self.lay2.removeItem(self.canvas2)

            self.ani.event_source.remove()
            self.ani2.event_source.remove()
            print('하하하')

            # plt.close(self.fig)
            # plt.close(self.fig2)

    # th.giveContour가 for 안에 있으므로, scatter_ImageProcess도 반복반복
    def scatter_ImageProcess(self, cnts):
        self.xdata = []
        self.ydata = []
        for a, b in enumerate(cnts):

            M = cv2.moments(b)
            (dcX, dcY), (width, height), _ = cv2.fitEllipse(b)
            parameter = cv2.arcLength(b, True)
            # print('width',width)
            # print('height',height)
            # contourArea < 150 하니까 거의 다 보임. 원래는 < 100
            # 초파리 중간 크기 WIdth = 24.07 px (장축)
            # <150. <25, <25 하니 상당히 정확함
            if cv2.contourArea(b) < 150 and width <= 25 and height <= 25 and parameter < 70:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                self.xdata.append(cX)
                self.ydata.append(cY)

    def windrose_ImageProcess(self, nowframe):

        if nowframe >= 1:
            self.xdata2 = []
            self.ydata2 = []
            self.coordinate_list = []
            self.initation.append(nowframe)
            # nowframe을 동결하고싶다.
            for a, b in enumerate(self.a[nowframe - 1]):

                M = cv2.moments(b)
                (dcX, dcY), (width, height), _ = cv2.fitEllipse(b)
                parameter = cv2.arcLength(b, True)
                if cv2.contourArea(b) < 150 and width <= 25 and height <= 25 and parameter < 70:
                    cX2 = int(M["m10"] / M["m00"])
                    cY2 = int(M["m01"] / M["m00"])
                    self.xdata2.append(cX2)
                    self.ydata2.append(cY2)

            # k번째 frame의 초파리 좌표들 [ [  ,  ], [  ,  ], ...] 꼴로 coordinate_list에 저장
            for k in list(zip(self.xdata2, self.ydata2)):
                q = list(k)
                self.coordinate_list.append(q)

                # print('coordinate_list', self.coordinate_list)
                # print('coordinate_list length',len(self.coordinate_list))

                # x y 랑 개수 역시 같게 나온다
            self.coordination = np.array(self.coordinate_list)
            # k 번째 frame의 초파리 좌표쌍들이 1차원 배열로 나열되어
            # **** 이놈이 이미 인접한 프레임의 컨투어 좌표들을 갖고있으면 좋겠다. ****
            # print('초파리 추정 좌표 수 갯수', len(self.coordination))

            ############## 왜 게속 초기화 되냐고 !!!!!!!!!!!###############################
            self.coordination_array.append(self.coordination)
            # print('coordination_array''s length', len(self.coordination_array))
            # print('시작하는 frame ',self.initation[0])
            # frame 1 - coordination_array[0]

            # print('c_array',self.coordination_array)

            # coordination array  - 각 frame의 초파리 contour좌표들을 모은 배열들의 집합배열
            # self. coordination_array = [ [[cX0,cY0], [cX1,cY1], ...[cXN,cYN]] , [[cX0',cY0'], [cX1',cY1'] , ... [cXN',cYN']], ....]

            self.chosen_xdata2 = []
            self.chosen_ydata2 = []

            self.scaled_chosen_xdata2 = []
            self.scaled_chosen_ydata2 = []
            # for k in range(len(self.coordination_array)):
            # if k==1: break
            a = np.zeros([len(self.xdata2), len(self.ydata2)])

            if len(self.coordination_array) >= 2:
                limit = min(len(self.coordination_array[nowframe - self.initation[0] - 1]),
                            len(self.coordination_array[nowframe - self.initation[0]]))

                # limit = 106
                # 0,1,2,3,4.....,105
                for i in range(limit):
                    for j in range(limit):
                        # if k == len(self.coordination_array)-1:
                        # if np.abs(len(self.coordination_array[nowframe-7]) - len(self.coordination_array[nowframe-6])) >= 10:
                        #    continu
                        # else:
                        a[i, j] = math.sqrt((self.coordination_array[nowframe - self.initation[0] - 1][i][0] -
                                             self.coordination_array[nowframe - self.initation[0]][j][0]) ** 2
                                            + (self.coordination_array[nowframe - self.initation[0] - 1][i][1] -
                                               self.coordination_array[nowframe - self.initation[0]][j][1]) ** 2)

            (r, c) = a.shape

            for i in range(r):
                for j in range(c):
                    if a[i][j] <= 2 and a[i][j] > 0:
                        self.chosen_xdata2.append(self.coordination_array[nowframe - self.initation[0] - 1][i][0])
                        self.chosen_ydata2.append(self.coordination_array[nowframe - self.initation[0] - 1][i][1])

            for k in self.chosen_xdata2:
                self.scaled_chosen_xdata2.append(k - w / 2)
            for k in self.chosen_ydata2:
                self.scaled_chosen_ydata2.append(k - h / 2)

            # print('chosenx', self.chosen_xdata2)
            # print('choseny', self.chosen_ydata2)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()