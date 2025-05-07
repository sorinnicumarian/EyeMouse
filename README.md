# Eyemouse - Head Direction Mouse Control

Eyemouse is a computer vision-based Python application that tracks a user's head pose using facial landmarks and approximates direction or tilt. The primary purpose is to allow rudimentary mouse control based on head orientation using a standard webcam.

## Features

- Face detection using Haar Cascade
- 68-point facial landmark detection with Dlib
- Real-time head angle estimation (horizontal tilt)
- Visual angle overlay and landmark rendering

## Requirements

- Python 3.x
- OpenCV
- Dlib
- shape_predictor_68_face_landmarks.dat (Dlib model)

## Installation

```bash
pip install opencv-python dlib numpy
```

Place the `shape_predictor_68_face_landmarks.dat` file in the project directory.

## Usage

```bash
python eyemouse.py
```

Press **Q** to quit the application.

## Example Output

Below is an example of what the interface looks like while tracking:

![Eyemouse Example Output](https://github.com/sorinnicumarian/EyeMouse/blob/main/Demo%20Screenshot.png)

The angle in degrees is displayed in real-time along with facial landmarks.

## Future Improvements

- Actual mouse control via `pyautogui` or `autopy`
- Vertical tilt and zoom gestures
- Enhanced stability and calibration
- Gaze estimation for finer control

---

© 2025 Sorin Nicu Marian — For educational and experimental use.
