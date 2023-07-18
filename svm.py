import cv2
import mediapipe as mp
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt
from daba.Mongo import collection
from sklearn.metrics.pairwise import cosine_similarity


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
        results = list(faces.get({}))  # Convert cursor to list to maintain order

        known_face_encodings = []
        known_face_names = []
        known_face_ids = []
        for result in results:
            known_face_encodings.append(np.array(result['Face']))
            known_face_names.append(result['Name'])
            known_face_ids.append(result['_id'])

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
            face_locations = [(landmark.x, landmark.y, landmark.z) for landmark in face_landmarks.landmark]
            face_embedding = np.asarray(face_locations).flatten()  # Convert to single numpy array

            if self.mode == 'recognition' and self.known_face_encodings:
                self.recognize_face(image, face_embedding, face_locations)
            elif self.mode == 'recording':
                self.record_face(face_embedding)

    def recognize_face(self, image, face_embedding, face_locations):
        max_similarity = -1
        name = "Unknown"

        for known_face_embedding, known_face_name in zip(self.known_face_encodings, self.known_face_names):
            distance = cosine_similarity([face_embedding], [known_face_embedding])
            if distance[0][0] > max_similarity:
                max_similarity = distance[0][0]
                name = known_face_name

        print(max_similarity, name)
        # If face is not close enough to any known faces, label as unknown
        if max_similarity < 0.999:
            name = "Unknown"

        # Convert landmarks to pixel coordinates and calculate bounding box
        pixel_landmarks = [(int(landmark[0] * image.shape[1]), int(landmark[1] * image.shape[0])) for landmark in
                           face_locations]
        left = min(landmark[0] for landmark in pixel_landmarks)
        right = max(landmark[0] for landmark in pixel_landmarks)
        top = min(landmark[1] for landmark in pixel_landmarks)
        bottom = max(landmark[1] for landmark in pixel_landmarks)

        face_width = right - left
        face_height = bottom - top

        # Calculate 10% of the face width and height
        width_increase = int(face_width * 0.10)
        height_increase = int(face_height * 0.10)

        # Make sure rectangle coordinates don't go outside the image
        left = max(0, left - width_increase)
        right = min(image.shape[1], right + width_increase)
        top = max(0, top - height_increase)
        bottom = min(image.shape[0], bottom + height_increase)

        # Draw a box around the face and a label with a name below the face
        cv2.rectangle(image, (left, top), (right, bottom), (96, 133, 29), 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (96, 133, 29), cv2.FILLED)
        cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

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
