import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing_utils = mp.solutions.drawing_utils

def calculate_eye_direction(image, landmarks):
    # Eye landmarks for left and right eye
    left_eye_outer = np.array([landmarks.landmark[130].x, landmarks.landmark[130].y]) * [image.shape[1], image.shape[0]]
    left_eye_inner = np.array([landmarks.landmark[133].x, landmarks.landmark[133].y]) * [image.shape[1], image.shape[0]]
    right_eye_outer = np.array([landmarks.landmark[359].x, landmarks.landmark[359].y]) * [image.shape[1], image.shape[0]]
    right_eye_inner = np.array([landmarks.landmark[362].x, landmarks.landmark[362].y]) * [image.shape[1], image.shape[0]]
    
    # Calculate vectors for each eye direction
    left_eye_vector = left_eye_outer - left_eye_inner
    right_eye_vector = right_eye_outer - right_eye_inner
    
    # Average the vectors to get a general direction of gaze
    average_eye_vector = (left_eye_vector + right_eye_vector) / 2.0
    
    # Calculate the angle of the gaze direction
    angle = np.degrees(np.arctan2(average_eye_vector[1], average_eye_vector[0]))
    if angle < 0:
        angle += 360  # Normalize angle to 0-360 degrees
    return angle

def get_head_direction(angle):
    return angle

def move_mouse_based_on_angle(angle, speed=10):
    # Convert angle to radians
    angle_rad = math.radians(angle)
    
    # Calculate vector components based on the angle
    x_component = math.cos(angle_rad) * speed
    y_component = -math.sin(angle_rad) * speed  # Negative because screen coordinates are inverted on the y-axis
    
    # Move the mouse relative to its current position
    pyautogui.moveRel(x_component, y_component)

cam = cv2.VideoCapture(0)

while True:
    success, frame = cam.read()
    if not success:
        break

    # frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            angle = calculate_eye_direction(frame, face_landmarks)
            direction_angle = get_head_direction(angle)
            print(f"Head Direction Angle: {direction_angle} degrees")
            
            # Move the mouse based on the calculated angle
            move_mouse_based_on_angle(direction_angle)

            drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()