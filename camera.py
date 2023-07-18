import cv2
import face_recognition
import dlib
import mediapipe as mp
import numpy
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
        faces = collection('Faces')
        results = faces.get({})

        known_face_encodings = []
        known_face_names = []
        for result in results:
            known_face_encodings.append(numpy.array(result['Face']))
            known_face_names.append(result['Name'])
        self.known_face_encodings = known_face_encodings if known_face_encodings else []
        self.known_face_names = known_face_names
        self.face_encoding = None
        self.username = None
        self.recognized_names = []
        self.prev_face_distance = float('inf')

    def run(self):
        # Instantiate the necessary components for the camera thread
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

        cap = cv2.VideoCapture(self.camera, cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 920)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)  # Adjust brightness (values can range from 0 to 1)
        cap.set(cv2.CAP_PROP_CONTRAST, 0.6)  # Adjust contrast (values can range from 0 to 1)
        cap.set(cv2.CAP_PROP_SATURATION, 0.6)  # Adjust saturation (values can range from 0 to 1)
        cap.set(cv2.CAP_PROP_FPS, 60)

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
            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=drawing_spec
            # )
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
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Use face_distance instead of compare_faces
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            # Only consider it a match if the distance is below a certain threshold (e.g., 0.4)
            best_match_index = np.argmin(face_distances)
            current_face_distance = face_distances[best_match_index]
            print(current_face_distance)
            if current_face_distance < 0.3:
                name = f"{self.known_face_names[best_match_index]}"
                if abs(current_face_distance - self.prev_face_distance) > 0.1:
                    # Reset the recognized names for new face
                    self.recognized_names = []
                self.recognized_names.append(name)
                most_common_name = max(set(self.recognized_names), key=self.recognized_names.count)
                name = most_common_name
            else:
                name = "Unknown"
                # self.recognized_names = []

            self.prev_face_distance = current_face_distance

            # Calculate the dimensions of the face rectangle
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
