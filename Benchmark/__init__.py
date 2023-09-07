from Benchmark.DeepFace import df
from Benchmark.face_recognition import fr
from Benchmark.MediaPipe import mp_reg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSlider, QFileDialog, QHBoxLayout
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from daba.Mongo import collection
from bson import ObjectId

accuracy = collection("Accuracy")

class ui(QWidget):
    cameraStarted = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.framework = ObjectId("64f9c40357014c27ea2bfa0d")
        self.camera_thread = None
        self.selected_folder = None
        self.title = 'Face Recognition AI - Benchmarking © Vishnu Prakash'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.min_detection_confidence = 0.8
        self.init_ui()
        self.known_face_encoding = None
        self.current_framework = mp_reg

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(640, 1000)

        self.layout = QVBoxLayout()

        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        self.recognized_face_label = QLabel("Recognised Face: ")
        self.layout.addWidget(self.recognized_face_label)

        # Dropdown for framework selection
        self.framework_dropdown = QComboBox()
        self.framework_dropdown.addItem("MediaPipe")
        self.framework_dropdown.addItem("DeepFace")
        self.framework_dropdown.addItem("Face_recognition")
        self.framework_dropdown.currentIndexChanged.connect(self.select_framework)

        # Folder Selection Button
        self.select_folder_button = QPushButton('Select Folder')
        self.select_folder_button.clicked.connect(self.select_folder)

        self.record_recognize_layout = QHBoxLayout()
        # Record and Recognize Buttons
        self.record_button = QPushButton('Record Face')
        self.record_button.clicked.connect(self.record_face)

        self.recognize_button = QPushButton('Recognise Face')
        self.recognize_button.clicked.connect(self.recognise_face)

        # Confidence Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(int(self.min_detection_confidence * 100))
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(5)
        self.slider.valueChanged.connect(self.set_min_detection_confidence)

        # Adding Widgets to Layout
        self.layout.addWidget(QLabel("Select Framework:"))
        self.layout.addWidget(self.framework_dropdown)
        self.layout.addWidget(self.select_folder_button)
        self.layout.addWidget(QLabel("Set Confidence %:"))
        self.layout.addWidget(self.slider)
        self.record_recognize_layout.addWidget(self.record_button)
        self.record_recognize_layout.addWidget(self.recognize_button)
        self.layout.addLayout(self.record_recognize_layout)

        self.setLayout(self.layout)

    def select_framework(self, index):
        if index == 0:  # If MediaPipe is selected
            self.framework = ObjectId("64f9c40357014c27ea2bfa0d")
            self.current_framework = mp_reg
        elif index == 1:  # If DeepFace is selected
            self.framework = ObjectId("64f9d166ff8a90f597fc5f95")
            self.current_framework = df
        elif index == 2:  # If Face_recognition is selected
            self.framework = ObjectId("64f9d179ff8a90f597fc5f97")
            self.current_framework = fr

    def select_folder(self):
        print(self.current_framework)
        folder = QFileDialog.getExistingDirectory(None, "Select Folder")
        if folder:
            self.selected_folder = folder
            print(f"Selected folder is {self.selected_folder}")

    @pyqtSlot()
    def record_face(self):
        self.camera_thread = self.current_framework(self.min_detection_confidence, mode='recording', selected_folder=self.selected_folder)
        self.camera_thread.changePixmap.connect(self.set_image)
        self.camera_thread.finished.connect(self.camera_stopped)
        self.camera_thread.start()
        self.cameraStarted.emit()

    @pyqtSlot()
    def recognise_face(self):
        self.camera_thread = self.current_framework(self.min_detection_confidence, selected_folder=self.selected_folder, mode='recognition')
        self.camera_thread.changePixmap.connect(self.set_image)
        self.camera_thread.faceRecognized.connect(self.update_recognized_face)
        self.camera_thread.displayButtons.connect(self.show_buttons)
        self.camera_thread.finished.connect(self.camera_stopped)
        self.camera_thread.start()
        self.cameraStarted.emit()

    @pyqtSlot(str)
    def update_recognized_face(self, name):
        self.recognized_face_label.setText(f"Recognised Face: {name}")

    @pyqtSlot()
    def camera_stopped(self):
        self.camera_thread = None

    def set_min_detection_confidence(self, value):
        self.min_detection_confidence = value / 100

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot()
    def show_buttons(self):
        self.button_layout = QHBoxLayout()
        # Create buttons and connect them to slots
        self.true_positive_button = QPushButton("True Positive", self)
        self.false_positive_button = QPushButton("False Positive", self)
        self.true_negative_button = QPushButton("True Negative", self)
        self.false_negative_button = QPushButton("False Negative", self)

        self.button_layout.addWidget(self.true_positive_button)
        self.button_layout.addWidget(self.false_positive_button)
        self.button_layout.addWidget(self.true_negative_button)
        self.button_layout.addWidget(self.false_negative_button)

        # ... (your existing code)
        self.layout.addLayout(self.button_layout)

        # Connect buttons to database updating functions
        self.true_positive_button.clicked.connect(self.update_database_true_positive)
        self.false_positive_button.clicked.connect(self.update_database_false_positive)
        self.true_negative_button.clicked.connect(self.update_database_true_negative)
        self.false_negative_button.clicked.connect(self.update_database_false_negative)

    def update_database_true_positive(self):
        data = {
            "_id": self.framework
        }
        accuracy.inc(data, {"TruePositive":1, "TotalCount":1})
        self.camera_thread.moveToNextImage.emit(True)
        self.remove_and_delete_buttons()

    def update_database_false_positive(self):
        data = {
            "_id": self.framework
        }
        accuracy.inc(data, {"FalsePositive": 1, "TotalCount": 1})
        self.camera_thread.moveToNextImage.emit(True)
        self.remove_and_delete_buttons()

    def update_database_true_negative(self):
        data = {
            "_id": self.framework
        }
        accuracy.inc(data, {"TrueNegative": 1, "TotalCount": 1})
        self.camera_thread.moveToNextImage.emit(True)
        self.remove_and_delete_buttons()

    def update_database_false_negative(self):
        data = {
            "_id": self.framework
        }
        accuracy.inc(data, {"FalseNegative": 1, "TotalCount": 1})
        self.camera_thread.moveToNextImage.emit(True)
        self.remove_and_delete_buttons()

    def remove_and_delete_buttons(self):
        self.button_layout.removeWidget(self.true_positive_button)
        self.button_layout.removeWidget(self.false_positive_button)
        self.button_layout.removeWidget(self.true_negative_button)
        self.button_layout.removeWidget(self.false_negative_button)

        self.true_positive_button.deleteLater()
        self.false_positive_button.deleteLater()
        self.true_negative_button.deleteLater()
        self.false_negative_button.deleteLater()