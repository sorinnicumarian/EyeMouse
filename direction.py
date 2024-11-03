import cv2
import mediapipe as mp
import pyautogui
import time
import math


cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_width, screen_height = pyautogui.size()

# Constants
SPEED = 30  # pixels per step
DELAY = 0.01  # delay between each step in seconds

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_height, frame_width, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 2, (0, 255, 0))

            if id == 1:
                screen_x = int(screen_width / frame_width * x)
                screen_y = int(screen_height / frame_height * y)

                current_x, current_y = pyautogui.position()
                # Calculate direction vector
                direction_x = screen_x - current_x
                direction_y = screen_y - current_y

                # Calculate magnitude of direction vector
                magnitude = math.sqrt(direction_x**2 + direction_y**2)

                # Normalize direction vector to get unit vector
                if magnitude > 0:
                    unit_x = direction_x / magnitude
                    unit_y = direction_y / magnitude
                else:
                    unit_x, unit_y = 0, 0

                # Move cursor in steps towards target
                while magnitude > SPEED:
                    current_x += unit_x * SPEED
                    current_y += unit_y * SPEED
                    pyautogui.moveTo(round(current_x), round(current_y))
                    time.sleep(DELAY)
                    
                    # Update direction vector and magnitude
                    direction_x = screen_x - current_x
                    direction_y = screen_y - current_y
                    magnitude = math.sqrt(direction_x**2 + direction_y**2)

                pyautogui.moveTo(screen_x, screen_y)

        left_eye = [landmarks[145], landmarks[159]]
        for landmark in left_eye:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 2, (0, 255, 255))
        # if (left_eye[0].y - left_eye[1].y) < 0.04:
        #     pyautogui.click()
        #     pyautogui.sleep(1)

            

    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.waitKey(1)

