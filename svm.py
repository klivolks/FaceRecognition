import cv2
import mediapipe as mp
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt
from daba.Mongo import collection
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.transform import Rotation as R
from sklearn import linear_model

PITCH = 'pitch'
ROLL = 'roll'
YAW = 'yaw'


class CameraThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, camera, min_detection_confidence, mode='recognition', known_face_encodings=None):
        super().__init__()

        # Initialize instance variables
        self.initial_source_points = None
        self.affine_transform = None
        self.target_points = np.zeros((468, 3))
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

            # Calculate pitch, roll, and yaw
            pitch, yaw, roll = self.calculate_pitch_yaw_roll(face_locations)

            self.target_points = np.array(face_locations)

            # Estimate the affine transformation matrix for this face
            self.affine_transform = self.estimate_affine_transform(np.array(face_locations), self.target_points)

            # Apply rotation and affine transformation to normalize the pose
            face_locations_transformed = self.apply_affine_transformation(face_locations, pitch, yaw, roll,
                                                                          self.affine_transform)

            face_embedding = np.asarray(face_locations_transformed).flatten()  # Convert to single numpy array

            if self.mode == 'recognition' and self.known_face_encodings:
                self.recognize_face(image, face_embedding, face_locations)
            elif self.mode == 'recording':
                self.record_face(face_embedding)

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
