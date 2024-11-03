import cv2
import dlib
import numpy as np

# Constants
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0, 255, 0)
LANDMARK_COLOR = (255, 0, 0)
LANDMARK_RADIUS = 2
ANGLE_DISPLAY_POSITION = (10, 30)
ANGLE_FONT_SCALE = 1
ANGLE_FONT_THICKNESS = 2

def load_predictor():
    try:
        return dlib.shape_predictor(PREDICTOR_PATH)
    except RuntimeError as e:
        print(f"Error loading shape predictor: {e}")
        return None

def load_face_cascade():
    try:
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier.")
        return face_cascade
    except Exception as e:
        print(f"Error loading Haar Cascade classifier: {e}")
        return None

def calculate_head_angle(landmarks):
    # Define points for calculations
    nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])
    chin = np.array([landmarks.part(8).x, landmarks.part(8).y])
    left_eye_corner = np.array([landmarks.part(36).x, landmarks.part(36).y])
    right_eye_corner = np.array([landmarks.part(45).x, landmarks.part(45).y])

    # Calculate center of the eyes and angle
    eye_center = (left_eye_corner + right_eye_corner) / 2.0
    delta_y, delta_x = nose_tip[1] - eye_center[1], nose_tip[0] - eye_center[0]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    return angle

def draw_landmarks(frame, landmarks):
    for n in range(68):
        x, y = landmarks.part(n).x, landmarks.part(n).y
        cv2.circle(frame, (x, y), LANDMARK_RADIUS, LANDMARK_COLOR, -1)

def main():
    predictor = load_predictor()
    face_cascade = load_face_cascade()
    
    if predictor is None or face_cascade is None:
        return

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmarks = predictor(gray, dlib_rect)
            angle = calculate_head_angle(landmarks)

            # Display angle
            cv2.putText(frame, f"Angle: {angle:.2f}", ANGLE_DISPLAY_POSITION, FONT, 
                        ANGLE_FONT_SCALE, TEXT_COLOR, ANGLE_FONT_THICKNESS)

            # Draw landmarks
            draw_landmarks(frame, landmarks)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()