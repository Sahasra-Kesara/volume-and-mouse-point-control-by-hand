from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

def detect_hands(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame, results

def change_volume(fingers_distance):
    vol = np.interp(fingers_distance, [30, 300], [min_vol, max_vol])
    volume.SetMasterVolumeLevel(vol, None)
    vol_percent = np.interp(vol, [min_vol, max_vol], [0, 100])
    print(f"Distance: {fingers_distance} -> Volume: {vol_percent}%")
    return vol_percent

def move_mouse(x, y):
    screen_width, screen_height = pyautogui.size()
    x = np.interp(x, [0, 640], [0, screen_width])
    y = np.interp(y, [0, 480], [0, screen_height])
    pyautogui.moveTo(x, y)
    print(f"Mouse moved to: ({x}, {y})")

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame, results = detect_hands(frame)
            vol_percent = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    h, w, c = frame.shape
                    thumb_tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))
                    index_finger_tip = (int(landmarks[8].x * w), int(landmarks[8].y * h))

                    # Calculate distance between thumb tip and index finger tip
                    distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_finger_tip))
                    vol_percent = change_volume(distance)

                    # Move mouse with index finger
                    move_mouse(index_finger_tip[0], index_finger_tip[1])

            # Display volume percentage on frame
            cv2.putText(frame, f'Volume: {int(vol_percent)}%', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
