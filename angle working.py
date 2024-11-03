import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing_utils = mp.solutions.drawing_utils

def calculate_head_angle(image, landmarks):
    # Convert MediaPipe landmark indices to numpy array
    nose_tip = np.array([landmarks.landmark[4].x, landmarks.landmark[4].y]) * [image.shape[1], image.shape[0]]
    chin = np.array([landmarks.landmark[152].x, landmarks.landmark[152].y]) * [image.shape[1], image.shape[0]]
    left_eye_corner = np.array([landmarks.landmark[226].x, landmarks.landmark[226].y]) * [image.shape[1], image.shape[0]]
    right_eye_corner = np.array([landmarks.landmark[446].x, landmarks.landmark[446].y]) * [image.shape[1], image.shape[0]]
    
    # Calculate the center of the eyes
    eye_center = (left_eye_corner + right_eye_corner) / 2.0
    
    # Calculate the angle between the eye center and nose tip
    delta_y = nose_tip[1] - eye_center[1]
    delta_x = nose_tip[0] - eye_center[0]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    
    return angle

cam = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculate the head angle
            angle = calculate_head_angle(frame, face_landmarks)
            print(f"Head Angle: {angle:.2f}")
            # Optionally draw landmarks on the face
            drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()