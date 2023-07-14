import dlib
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSlider
from PyQt6.QtCore import Qt
import face_recognition
import cv2
import mediapipe as mp
import numpy as np


class CameraThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, camera, min_detection_confidence, mode='recognition', known_face_encoding=None):
        super(CameraThread, self).__init__()
        self.camera = camera
        self.min_detection_confidence = min_detection_confidence
        self.running = False
        self.mode = mode
        self.known_face_encoding = known_face_encoding
        self.face_encoding = None

    def run(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

        cap = cv2.VideoCapture(self.camera, cv2.CAP_ANY)
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   min_detection_confidence=self.min_detection_confidence) as face_mesh:
            self.running = True
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = face_mesh.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=drawing_spec
                            )
                            # Resize frame of video to 1/4 size for faster face recognition processing
                            small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

                            # Detect the faces in the small frame
                            face_locations = face_recognition.face_locations(small_frame)
                            face_landmarks = []
                            for (top, right, bottom, left) in face_locations:
                                rect = dlib.rectangle(left, top, right, bottom)
                                shape = predictor(small_frame, rect)
                                face_landmarks.append(shape)

                            # Now we will encode faces using face_landmarks instead of using face_recognition.face_encodings
                            face_encodings = [np.array(face_recognition_model.compute_face_descriptor(small_frame, face_landmark, 1)) for face_landmark in face_landmarks]

                            if self.mode == 'recognition' and self.known_face_encoding is not None and len(face_encodings) > 0:
                                # See if the face is a match for the known face(s)
                                matches = face_recognition.compare_faces([self.known_face_encoding], face_encodings[0])
                                name = "Recognized" if True in matches else "Unknown"

                                # Draw a box around the face
                                cv2.rectangle(image, (left*4, top*4), (right*4, bottom*4), (0, 0, 255), 2)

                                # Draw a label with a name below the face
                                cv2.rectangle(image, (left*4, bottom*4 - 35), (right*4, bottom*4), (0, 0, 255), cv2.FILLED)
                                font = cv2.FONT_HERSHEY_DUPLEX
                                cv2.putText(image, name, (left*4 + 6, bottom*4 - 6), font, 1.0, (255, 255, 255), 1)

                            elif self.mode == 'recording' and len(face_encodings) > 0:
                                # For simplicity, consider only the first face detected
                                self.face_encoding = face_encodings[0]

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
        self.known_face_encoding = None

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

    @pyqtSlot()
    def record_face(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop_camera()
            self.camera_thread = None

        self.camera_thread = CameraThread(self.current_camera, self.min_detection_confidence, mode='recording')
        self.camera_thread.changePixmap.connect(self.set_image)
        self.camera_thread.finished.connect(self.face_recorded)
        self.camera_thread.start()
        self.cameraStarted.emit()

    @pyqtSlot()
    def face_recorded(self):
        if self.camera_thread:
            self.known_face_encoding = self.camera_thread.face_encoding
        self.camera_thread = None

    @pyqtSlot()
    def recognize_face(self):
        if self.known_face_encoding is None:
            print("No face recorded yet")
            return

        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop_camera()
            self.camera_thread = None

        self.camera_thread = CameraThread(self.current_camera, self.min_detection_confidence, mode='recognition',
                                          known_face_encoding=self.known_face_encoding)
        self.camera_thread.changePixmap.connect(self.set_image)
        self.camera_thread.finished.connect(self.camera_stopped)
        self.camera_thread.start()
        self.cameraStarted.emit()

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == '__main__':
    app = QApplication([])
    ex = CameraSelector()
    ex.show()
    app.exec()
