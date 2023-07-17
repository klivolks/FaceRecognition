import cv2
import dlib
import mediapipe as mp
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt
from daba.Mongo import collection
from sklearn import svm


class CameraThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, camera, min_detection_confidence, mode='recognition', known_face_encodings=None):
        super().__init__()

        # Initialize instance variables
        self.camera = camera
        self.min_detection_confidence = min_detection_confidence
        self.running = False
        self.mode = mode

        # Initialize face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

        # Retrieve faces from database and train SVM classifier
        faces = collection('Faces')
        results = faces.get({})

        face_encodings = []
        face_names = []
        for result in results:
            face_encodings.append(np.array(result['Face']))
            face_names.append(result['Name'])

        # Train classifier
        self.clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
        self.clf.fit(face_encodings, face_names)

    def run(self):
        # Instantiate the necessary components for the camera thread
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        cap = cv2.VideoCapture(self.camera, cv2.CAP_ANY)

        # Start the face detection
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   min_detection_confidence=self.min_detection_confidence) as face_mesh:
            self.running = True

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue

                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    self.process_faces(results, image)

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.changePixmap.emit(p)

        cap.release()

    def process_faces(self, results, image):
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # Convert image to gray scale for dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector(gray)

        for face in faces:
            # Get landmarks
            shape = self.predictor(gray, face)

            # Get face encoding
            face_descriptor = self.face_rec_model.compute_face_descriptor(image, shape)

            # Predict face label
            face_label = self.clf.predict(np.array([np.array(face_descriptor)]))[0]

            # Draw box and label around face
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, face_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    @pyqtSlot()
    def stop_camera(self):
        self.running = False
        self.quit()
        self.wait()
