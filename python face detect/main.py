import cv2
import time
import numpy as np
import pyttsx3
from datetime import timedelta
from threading import Thread
import mediapipe as mp

# Text-to-speech setup
engine = pyttsx3.init()
def speak(text):
    Thread(target=engine.say, args=(text,)).start()
    engine.runAndWait()

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Thresholds
EYE_CLOSED_THRESHOLD = 0.2
SLEEP_WARNING_SECONDS = 5

# Eye aspect ratio function
def eye_aspect_ratio(landmarks, eye_indices):
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[1]]
    top = landmarks[eye_indices[2]]
    bottom = landmarks[eye_indices[3]]

    hor_length = np.linalg.norm(np.array(left) - np.array(right))
    ver_length = np.linalg.norm(np.array(top) - np.array(bottom))

    if hor_length == 0:
        return 0
    return ver_length / hor_length

# Eye landmark indices (left and right)
LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]

cap = cv2.VideoCapture(0)

last_seen = time.time()
total_work_time = 0
sleep_start = None
awake = True
start_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    h, w, _ = img.shape

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_CLOSED_THRESHOLD:
            if sleep_start is None:
                sleep_start = time.time()
            elif time.time() - sleep_start > SLEEP_WARNING_SECONDS and awake:
                speak("Wake up! You've been inactive for a while.")
                awake = False
        else:
            if sleep_start:
                sleep_start = None
            if not awake:
                awake = True
            total_work_time += time.time() - last_seen
            last_seen = time.time()

        status = "Awake" if awake else "Sleeping"
    else:
        status = "No face detected"
        sleep_start = None
        awake = False

    # Display overlay
    work_duration = str(timedelta(seconds=int(total_work_time)))
    cv2.putText(img, f"Status: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if awake else (0, 0, 255), 2)
    cv2.putText(img, f"Active Work Time: {work_duration}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Work Monitor", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
