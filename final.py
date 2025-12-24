import cv2
import numpy as np
import onnxruntime as ort
import math
import time
from ultralytics import YOLO
import os

ALERT_IMAGE_PATH = "alert.png"
EMOTION_MODEL = r"C:\Users\hp\Downloads\your_model.onnx"

EMOTIONS = ["fear","neutral","happy","sad","anger","disgust","surprise","contempt"]
NEGATIVE_EMOTIONS = {"fear","disgust","surprise"}

HAND_NEAR_THRESHOLD = 60  

emotion_sess = ort.InferenceSession(EMOTION_MODEL, providers=["CPUExecutionProvider"])
emotion_in = emotion_sess.get_inputs()[0].name
emotion_out = emotion_sess.get_outputs()[0].name

pose_model = YOLO("yolov8n-pose.pt")

alert_img = cv2.imread(ALERT_IMAGE_PATH) if os.path.exists(ALERT_IMAGE_PATH) else None

def check_hand_near_neck(kp_xy, kp_conf):
    LS, RS = 5, 6   
    LW, RW = 9, 10  

    if kp_conf[LS] < 0.3 or kp_conf[RS] < 0.3:
        return False

    neck_x = (kp_xy[LS][0] + kp_xy[RS][0]) / 2
    neck_y = (kp_xy[LS][1] + kp_xy[RS][1]) / 2

    hands = []
    if kp_conf[LW] > 0.3:
        hands.append(kp_xy[LW])
    if kp_conf[RW] > 0.3:
        hands.append(kp_xy[RW])

    for hx, hy in hands:
        if math.dist((hx, hy), (neck_x, neck_y)) < HAND_NEAR_THRESHOLD:
            return True
    return False
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

asthma_alert = False

print("✔ Webcam Asthma Detection Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    e_in = cv2.resize(gray, (64,64)).astype(np.float32)/255.0
    e_in = e_in.reshape(1,64,64,1)
    out = emotion_sess.run([emotion_out], {emotion_in: e_in})
    logits = out[0][0]
    probs = np.exp(logits)/np.sum(np.exp(logits))
    e_idx = int(np.argmax(probs))
    emotion = EMOTIONS[e_idx]

    cv2.putText(frame, f"Emotion: {emotion}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    negative_emotion = emotion.lower() in NEGATIVE_EMOTIONS
    hand_alert = False
    if negative_emotion:
        results = pose_model(frame, verbose=False)
        if results[0].keypoints is not None:
            kp_xy = results[0].keypoints.xy[0].cpu().numpy()
            kp_conf = results[0].keypoints.conf[0].cpu().numpy()
            for i, (x, y) in enumerate(kp_xy):
                if kp_conf[i] > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)
            hand_alert = check_hand_near_neck(kp_xy, kp_conf)

    asthma_alert = negative_emotion and hand_alert

    if asthma_alert and alert_img is not None:
        display = cv2.resize(alert_img, (frame.shape[1], frame.shape[0]))
        cv2.putText(display, "!!! ASTHMA ALERT !!!", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    else:
        display = frame.copy()

    cv2.imshow("Asthma AI", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
