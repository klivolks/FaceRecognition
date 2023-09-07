import cv2
import os
import mediapipe as mp
import face_recognition
import dlib
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, QWaitCondition, QMutex
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt
from daba.Mongo import collection
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.transform import Rotation as R
from sklearn import linear_model
import time

PITCH = 'pitch'
ROLL = 'roll'
YAW = 'yaw'


class fr(QThread):
    displayButtons = pyqtSignal()
    moveToNextImage = pyqtSignal(bool)
    changePixmap = pyqtSignal(QImage)
    faceRecognized = pyqtSignal(str)

    def __init__(self, min_detection_confidence, mode='recognition', known_face_encodings=None, selected_folder=None):
        super().__init__()

        # Initialize instance variables
        self.selected_folder = selected_folder
        self.initial_source_points = None
        self.affine_transform = None
        self.target_points = np.zeros((468, 3))
        self.min_detection_confidence = min_detection_confidence
        self.running = False
        self.mode = mode
        faces = collection('Face_recognition')
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
        self.wait_condition = QWaitCondition()
        self.mutex = QMutex()
        self.moveToNextImage.connect(self.set_move_to_next)
        self.prev_face_distance = float('inf')

    @pyqtSlot(bool)
    def set_move_to_next(self, val):
        self.mutex.lock()
        if val:
            self.wait_condition.wakeAll()
        self.mutex.unlock()

    def run(self):
        start_time = round(time.time()*1000)
        print("running")
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   min_detection_confidence=self.min_detection_confidence) as face_mesh:
            print(self.selected_folder)
            if self.selected_folder:
                for subfolder in os.listdir(self.selected_folder):
                    subfolder_path = os.path.join(self.selected_folder, subfolder)
                    if os.path.isdir(subfolder_path):
                        for filename in os.listdir(subfolder_path):
                            if filename.endswith(('.jpg', '.png')):
                                image_path = os.path.join(subfolder_path, filename)
                                frame = cv2.imread(image_path)
                                if frame is not None:

                                    image = cv2.cvtColor(frame, 1)
                                    image.flags.writeable = False
                                    results = face_mesh.process(image)
                                    image.flags.writeable = True
                                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                                    if results.multi_face_landmarks:
                                        self.username = subfolder
                                        self.process_faces(results, image, predictor, face_recognition_model)

                                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    h, w, ch = rgb_image.shape
                                    bytes_per_line = ch * w
                                    convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                                    p = convert_to_qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                                    self.changePixmap.emit(p)
                                    end_time = round(time.time() * 1000)
                                    time_taken = end_time - start_time
                                    speed = collection('execution_time')
                                    speed.put({"framework": "Face_recognition", "Time": time_taken, "Process": self.mode})
            else:
                print("No folder selected")

    def process_faces(self, results, image, predictor, face_recognition_model):
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        for face_landmarks in results.multi_face_landmarks:
            small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
            face_locations = face_recognition.face_locations(small_frame)
            face_landmarks = [predictor(small_frame, dlib.rectangle(*face)) for face in face_locations]
            face_encodings = [np.array(face_recognition_model.compute_face_descriptor(small_frame, face_landmark, 1))
                              for face_landmark in face_landmarks]

            if self.mode == 'recognition' and self.known_face_encodings is not None and face_encodings:
                self.recognize_face(image, face_encodings, face_locations)

            elif self.mode == 'recording' and face_encodings:
                self.record_face(face_encodings)

    @staticmethod
    def calculate_pitch_yaw_roll(face_landmarks):
        # Convert the landmarks to a numpy array
        landmarks_array = np.array(face_landmarks)

        # Calculate the center of mass of the landmarks
        center_of_mass = landmarks_array.mean(axis=0)

        # Translate the landmarks to the origin
        landmarks_centered = landmarks_array - center_of_mass

        # Perform singular value decomposition on the covariance matrix of the centered landmarks
        u, s, vh = np.linalg.svd(landmarks_centered)

        # The columns of vh are the eigenvectors of the covariance matrix, so the rotation matrix is just vh
        rotation_matrix = vh

        # Convert the rotation matrix to Euler angles (pitch, yaw, roll)
        rotation = R.from_matrix(rotation_matrix)
        pitch, roll, yaw = rotation.as_euler('xyz', degrees=True)

        return pitch, yaw, roll

    @staticmethod
    def apply_affine_transformation(face_landmarks, pitch, yaw, roll, affine_transform):
        # Convert the landmarks to a numpy array
        landmarks_array = np.array(face_landmarks)

        # Create a rotation matrix from the Euler angles
        rotation = R.from_euler('xyz', [pitch, roll, yaw], degrees=True)
        rotation_matrix = rotation.as_matrix()

        # Apply the rotation matrix to the landmarks
        landmarks_rotated = np.dot(landmarks_array, rotation_matrix)

        # Prepare for the affine transformation
        landmarks_rotated = np.hstack(
            [landmarks_rotated, np.ones((landmarks_rotated.shape[0], 1))])  # append ones column

        # Apply the affine transformation
        landmarks_transformed = np.dot(landmarks_rotated, affine_transform.T)

        return landmarks_transformed

    @staticmethod
    def estimate_affine_transform(points_source, points_target):
        """
        Estimate affine transform from source points to target points.
        """
        # Check the shape of points_target as well
        assert points_target.shape[1] == 3, "points_target must have 3 columns"
        points_target = np.array(points_target)

        # Estimate affine transform
        model = linear_model.LinearRegression().fit(points_source[:, :3],
                                                    points_target)  # Use only the first three columns

        affine_transform = np.hstack([model.coef_, model.intercept_.reshape(-1, 1)])

        return affine_transform

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
            self.faceRecognized.emit(name)
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
            self.displayButtons.emit()
            self.mutex.lock()
            self.wait_condition.wait(self.mutex)
            self.mutex.unlock()

    def record_face(self, face_embedding):
        self.face_encoding = face_embedding[0]
        faces = collection('Face_recognition')
        data = {
            "Face": self.face_encoding.tolist(),
            "Name": self.username
        }
        faces.put(data)