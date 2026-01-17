import cv2
import numpy as np
import onnxruntime as ort
import math
import time
from ultralytics import YOLO
import os
import threading
import serial
ALERT_IMAGE_PATH = "/home/kfakh/Desktop/Ai_Model.2/alert.png"
EMOTION_MODEL = "/home/kfakh/Desktop/Ai_Model.2/your_model.onnx"

EMOTIONS = ["fear","neutral","happy","sad","anger","disgust","surprise","contempt"]
NEGATIVE_EMOTIONS = {"fear","disgust","sad","anger"}

HAND_NEAR_THRESHOLD = 60
ALERT_DURATION = 2.0  
POSITIVE_FREEZE_FRAMES = 100  
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
last_frame = None
countdown_elapsed = 0.0
last_tick_time = None
positive_frame_count = 0
asthma_alert = False
alert_sent = False

emotion_sess = ort.InferenceSession(EMOTION_MODEL, providers=["CPUExecutionProvider"])
emotion_in = emotion_sess.get_inputs()[0].name
emotion_out = emotion_sess.get_outputs()[0].name

pose_model = YOLO("yolov8n-pose.onnx")

alert_img = cv2.imread(ALERT_IMAGE_PATH) if os.path.exists(ALERT_IMAGE_PATH) else None

try:
    robot_serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("✔ Robot serial connected")
except:
    robot_serial = None
    print("✖ Serial connection failed")

def grab_last_frame(cap):
    global last_frame
    while True:
        ret, frame = cap.read()
        if ret:
            last_frame = frame.copy()

def check_hand_near_neck(kp_xy, kp_conf):
    LS, RS = 5, 6
    LW, RW = 9, 10

    if kp_conf[LS] < 0.3 or kp_conf[RS] < 0.3:
        return False

    neck_x = (kp_xy[LS][0] + kp_xy[RS][0]) / 2
    neck_y = (kp_xy[LS][1] + kp_xy[RS][1]) / 2

    for idx in [LW, RW]:
        if kp_conf[idx] > 0.3:
            hx, hy = kp_xy[idx]
            if math.dist((hx, hy), (neck_x, neck_y)) < HAND_NEAR_THRESHOLD:
                return True
    return False
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

threading.Thread(target=grab_last_frame, args=(cap,), daemon=True).start()

print("✔ Webcam Asthma Detection Started (Freeze/Resume Logic Enabled)")
while True:
    if last_frame is None:
        continue

    frame = last_frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    e_in = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
    e_in = e_in.reshape(1, 64, 64, 1)

    probs = emotion_sess.run([emotion_out], {emotion_in: e_in})[0][0]
    emotion = EMOTIONS[int(np.argmax(probs))]
    negative_emotion = emotion.lower() in NEGATIVE_EMOTIONS
    hand_near = False
    if negative_emotion:
        results = pose_model(frame, verbose=False)
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kp_xy = results[0].keypoints.xy[0].cpu().numpy()
            kp_conf = results[0].keypoints.conf[0].cpu().numpy()

            for i, (x, y) in enumerate(kp_xy):
                if kp_conf[i] > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            hand_near = check_hand_near_neck(kp_xy, kp_conf)

    now = time.time()
    if negative_emotion and hand_near:
        positive_frame_count = 0

        if last_tick_time is None:
            last_tick_time = now

        countdown_elapsed += now - last_tick_time
        last_tick_time = now

        if countdown_elapsed >= ALERT_DURATION:
            asthma_alert = True
            if not alert_sent and robot_serial is not None:
                robot_serial.write(b"ASTHMA_DETECTED\n")
                print("📤 ASTHMA_DETECTED sent to ESP")
                alert_sent = True
    else:
        last_tick_time = None

        if not negative_emotion:
            positive_frame_count += 1
            if positive_frame_count >= POSITIVE_FREEZE_FRAMES:
                countdown_elapsed = 0.0
                asthma_alert = False
                alert_sent = False
        else:
            countdown_elapsed = 0.0
            asthma_alert = False
            alert_sent = False
            positive_frame_count = 0
    display = frame.copy()

    if asthma_alert and alert_img is not None:
        overlay = cv2.resize(alert_img, (display.shape[1], display.shape[0]))
        display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
        cv2.putText(display, "!!! ASTHMA ALERT !!!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    if negative_emotion and hand_near and not asthma_alert:
        time_left = max(0, ALERT_DURATION - countdown_elapsed)
        cv2.putText(display, f"Validating: {time_left:.1f}s",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(display, f"Emotion: {emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(display, f"Negative: {negative_emotion}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(display, f"Hand Near Neck: {hand_near}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(display, f"Positive Frames: {positive_frame_count}",
                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    cv2.imshow("Asthma AI", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
if robot_serial is not None:
    robot_serial.close()
