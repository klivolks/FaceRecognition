import cv2
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSlider
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from camera import CameraThread
from record import RecordFace
from recognise import RecognizeFace


class CameraSelector(QWidget, RecordFace, RecognizeFace):
    cameraStarted = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.title = 'Camera Selector'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.min_detection_confidence = 0.8
        self.current_camera = 0
        self.camera_thread = None
        self.init_ui()
        self.populate_camera_dropdown()
        self.known_face_encoding = None

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(640, 1000)

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

        self.record_button = QPushButton('Record Face')
        self.record_button.clicked.connect(self.record_face)

        self.recognize_button = QPushButton('Recognize Face')
        self.recognize_button.clicked.connect(self.recognize_face)

        self.layout.addWidget(self.record_button)
        self.layout.addWidget(self.recognize_button)

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
        else:
            print(self.camera_thread)

