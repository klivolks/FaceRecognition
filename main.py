from PyQt6.QtCore import pyqtSignal, pyqtSlot, QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSlider
from PyQt6.QtCore import Qt
import cv2
import mediapipe as mp
import numpy as np


class CameraThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, camera, min_detection_confidence):
        super(CameraThread, self).__init__()
        self.camera = camera
        self.min_detection_confidence = min_detection_confidence
        self.running = False

    def run(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_face_detection = mp.solutions.face_detection

        cap = cv2.VideoCapture(self.camera, cv2.CAP_ANY)
        with mp_face_detection.FaceDetection(model_selection=1,
                                             min_detection_confidence=self.min_detection_confidence) as face_detection:
            self.running = True
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = face_detection.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.detections:
                        for detection in results.detections:
                            mp_drawing.draw_detection(image, detection)

                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    p = convert_to_qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                    self.changePixmap.emit(p)

        cap.release()

    @pyqtSlot()
    def stop_camera(self):
        self.running = False
        self.quit()
        self.wait()


class CameraSelector(QWidget):
    cameraStarted = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.title = 'Camera Selector'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.min_detection_confidence = 0.5
        self.current_camera = 0
        self.camera_thread = None
        self.init_ui()
        self.populate_camera_dropdown()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1000, 1000)

        self.layout = QVBoxLayout()

        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        self.dropdown = QComboBox()
        self.dropdown.currentIndexChanged.connect(self.select_camera)

        self.start_button = QPushButton('Start Camera')
        self.start_button.clicked.connect(self.start_camera)

        self.stop_button = QPushButton('Stop Camera')
        self.stop_button.clicked.connect(self.stop_camera)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(int(self.min_detection_confidence * 100))
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(5)
        self.slider.valueChanged.connect(self.set_min_detection_confidence)

        self.layout.addWidget(QLabel("Select Camera:"))
        self.layout.addWidget(self.dropdown)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)
        self.layout.addWidget(QLabel("Set Confidence %:"))
        self.layout.addWidget(self.slider)

        self.setLayout(self.layout)

    def populate_camera_dropdown(self):
        index = 0
        while True:
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
            if not cap.isOpened():
                break
            cap.release()
            self.dropdown.addItem(str(index))
            index += 1

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def select_camera(self, i):
        self.current_camera = i

    def set_min_detection_confidence(self, value):
        self.min_detection_confidence = value / 100

    @pyqtSlot()
    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop_camera()
            self.camera_thread = None

        self.camera_thread = CameraThread(self.current_camera, self.min_detection_confidence)
        self.camera_thread.changePixmap.connect(self.set_image)
        self.camera_thread.finished.connect(self.camera_stopped)
        self.camera_thread.start()
        self.cameraStarted.emit()

    @pyqtSlot()
    def camera_stopped(self):
        self.camera_thread = None

    @pyqtSlot()
    def stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop_camera()
            self.camera_thread.wait()

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == '__main__':
    app = QApplication([])
    ex = CameraSelector()
    ex.show()
    app.exec()
