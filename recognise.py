import numpy
from PyQt6.QtCore import pyqtSlot
from svm import CameraThread
from daba.Mongo import collection


class RecognizeFace:
    def __init__(self):
        self.camera_thread = None
        self.known_face_encoding = None

    @pyqtSlot()
    def recognize_face(self):
        faces = collection('Faces')
        results = faces.get({})

        known_face_encodings = []
        known_face_names = []
        for result in results:
            known_face_encodings.append(numpy.array(result['Face']))
            known_face_names.append(result['Name'])

        if not known_face_encodings:
            print("No face recorded yet")
            return

        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop_camera()
            self.camera_thread = None

        self.camera_thread = CameraThread(self.current_camera, self.min_detection_confidence, mode='recognition',
                                          known_face_encodings=known_face_encodings)
        self.camera_thread.changePixmap.connect(self.set_image)
        self.camera_thread.known_face_names = known_face_names
        self.camera_thread.finished.connect(self.camera_stopped)
        self.camera_thread.start()
        self.cameraStarted.emit()
