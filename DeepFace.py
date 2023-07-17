import cv2
import dlib
import mediapipe as mp
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt
from daba.Mongo import collection
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace


class CameraThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, camera, min_detection_confidence, mode='recognition', known_face_encodings=None):
        super().__init__()

        # Initialize instance variables
        self.camera = camera
        self.min_detection_confidence = min_detection_confidence
        self.running = False
        self.mode = mode
        faces = collection('Faces')
        results = faces.get({})

        known_face_encodings = []
        known_face_names = []
        for result in results:
            known_face_encodings.append(np.array(result['Face']))
            known_face_names.append(result['Name'])
        self.known_face_encodings = known_face_encodings if known_face_encodings else []
        self.known_face_names = known_face_names
        self.face_encoding = None
        self.username = None
        self.recognized_names = []

    def run(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        cap = cv2.VideoCapture(self.camera, cv2.CAP_ANY)

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

        for face_landmarks in results.multi_face_landmarks:
            face_locations = [[landmark.x, landmark.y] for landmark in face_landmarks.landmark]

            # Face recognition with DeepFace
            face_embedding = DeepFace.represent(image, model_name='VGG-Face', enforce_detection=False)

            if self.mode == 'recognition' and self.known_face_encodings:
                self.recognize_face(image, face_embedding, face_locations)
            elif self.mode == 'recording':
                self.record_face(face_embedding)

    def recognize_face(self, image, face_embedding, face_locations):
        min_distance = 10000  # initialize with a high number
        name = "Unknown"

        for known_face_embedding, known_face_name in zip(self.known_face_encodings, self.known_face_names):
            distance = cosine_similarity([face_embedding], [known_face_embedding])

            if distance < min_distance:
                min_distance = distance
                name = known_face_name

        # If face is not close enough to any known faces, label as unknown
        if min_distance > 0.4:
            name = "Unknown"

        # Calculate the dimensions of the face rectangle
        for (top, right, bottom, left) in face_locations:
            face_width = right - left
            face_height = bottom - top

            # Calculate 50% of the face width and height
            width_increase = int(face_width * 0.25)
            height_increase = int(face_height * 0.25)

            # Draw a box around the face and a label with a name below the face
            cv2.rectangle(image, ((left - width_increase) * 4, (top - height_increase) * 4),
                          ((right + width_increase) * 4, (bottom + height_increase) * 4), (96, 133, 29), 2)
            cv2.rectangle(image, ((left - width_increase) * 4, (bottom + height_increase) * 4 - 35),
                          ((right + width_increase) * 4, (bottom + height_increase) * 4), (96, 133, 29), cv2.FILLED)
            cv2.putText(image, name, ((left - width_increase) * 4 + 6, (bottom + height_increase) * 4 - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    def record_face(self, face_embedding):
        self.face_encoding = face_embedding
        faces = collection('Faces')
        data = {
            "Face": self.face_encoding.tolist(),
            "Name": self.username
        }
        faces.put(data)

    @pyqtSlot()
    def stop_camera(self):
        self.running = False
        self.quit()
        self.wait()
