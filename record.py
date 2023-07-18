from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QInputDialog
from svm import CameraThread


class RecordFace:
    def __init__(self):
        self.camera_thread = None
        self.known_face_encoding = None


    @pyqtSlot()
    def record_face(self):
        # Open a QInputDialog to get the name
        text, ok = QInputDialog.getText(self, 'Input Dialog', 'Enter your name:')

        if ok:
            username = str(text)
        else:
            return  # User closed the input dialog, do not proceed further

        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop_camera()
            self.camera_thread = None

        self.camera_thread = CameraThread(self.current_camera, self.min_detection_confidence, mode='recording')
        self.camera_thread.changePixmap.connect(self.set_image)
        self.camera_thread.finished.connect(self.face_recorded)
        self.camera_thread.start()

        # Pass the name entered by the user to the CameraThread
        self.camera_thread.username = username

        self.cameraStarted.emit()

    @pyqtSlot()
    def face_recorded(self):
        if self.camera_thread:
            self.known_face_encoding = self.camera_thread.face_encoding
        self.camera_thread = None
