import cv2
import face_recognition
import dlib
import mediapipe as mp
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt
from daba.Mongo import collection


class CameraThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, camera, min_detection_confidence, mode='recognition', known_face_encodings=None):
        super().__init__()

        # Initialize instance variables
        self.camera = camera
        self.min_detection_confidence = min_detection_confidence
        self.running = False
        self.mode = mode
        self.known_face_encodings = known_face_encodings if known_face_encodings else []
        self.known_face_names = []
        self.face_encoding = None
        self.username = None

    def run(self):
        # Instantiate the necessary components for the camera thread
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

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
                    self.process_faces(results, image, predictor, face_recognition_model)

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.changePixmap.emit(p)

        cap.release()

    def process_faces(self, results, image, predictor, face_recognition_model):
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )

            small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
            face_locations = face_recognition.face_locations(small_frame)
            face_landmarks = [predictor(small_frame, dlib.rectangle(*face)) for face in face_locations]
            face_encodings = [np.array(face_recognition_model.compute_face_descriptor(small_frame, face_landmark, 1))
                              for face_landmark in face_landmarks]

            if self.mode == 'recognition' and self.known_face_encodings is not None and face_encodings:
                self.recognize_face(image, face_encodings, face_locations)

            elif self.mode == 'recording' and face_encodings:
                self.record_face(face_encodings)

    def recognize_face(self, image, face_encodings, face_locations):
        # Compare each face in frame to known faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            count_true = sum(matches)  # Sums up all True values (as 1 each)
            count_false = len(matches) - count_true  # Subtracts sum of True from total length

            print(f"Total: {len(matches)}, Match: {count_true}, No Match: {count_false}")
            if False in matches:
                wrong = matches.index(False)
                print(f"Not face: {self.known_face_names[wrong]}")
            name = "Unknown"

            # Find the best match
            if True in matches:
                match_index = matches.index(True)
                name = f"Recognized: {self.known_face_names[match_index]}"

            # Draw a box around the face and a label with a name below the face
            cv2.rectangle(image, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)
            cv2.rectangle(image, (left * 4, bottom * 4 - 35), (right * 4, bottom * 4), (0, 0, 255), cv2.FILLED)
            cv2.putText(image, name, (left * 4 + 6, bottom * 4 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    def record_face(self, face_encodings):
        self.face_encoding = face_encodings[0]  # Consider only the first face detected
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
