"""
predict_camera_mediapipe.py
---------------------------------
Realtime Sign Language Recognition (IP Webcam)
- Camera: IP Webcam (Android)
- URL: http://10.0.0.50:8080/video
- MediaPipe Hands (multi-hand)
- MobileNetV2
- FPS + confidence smoothing
"""

import cv2
import time
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

# ================= CONFIG =================
MODEL_PATH = "mobilenetv2_sign_language.h5"
CLASS_NAME_PATH = "class_names.txt"

IP_CAMERA_URL = "http://192.168.2.22:8080/video"

IMG_SIZE = 224
CONF_THRESHOLD = 0.5
SMOOTHING_WINDOW = 7
MAX_HANDS = 2
# ==========================================


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def preprocess_hand(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def main():
    # ===== LOAD MODEL =====
    print("🔄 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names(CLASS_NAME_PATH)
    print("✅ Model loaded")
    print("📌 Classes:", class_names)

    # ===== MediaPipe Hands =====
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # ===== OPEN IP CAMERA =====
    print("🔌 Connecting to IP Webcam...")
    cap = cv2.VideoCapture(IP_CAMERA_URL)

    if not cap.isOpened():
        print("❌ Không kết nối được IP Webcam")
        print("👉 Kiểm tra:")
        print("   - Điện thoại & PC cùng mạng WiFi")
        print("   - IP Webcam đang chạy")
        print("   - URL đúng: http://10.0.0.50:8080")
        return

    print("🎥 IP Camera connected – Nhấn 'q' để thoát")

    prev_time = 0
    pred_buffer = deque(maxlen=SMOOTHING_WINDOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Mất kết nối camera")
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # ===== BOUNDING BOX =====
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]

                x_min = int(min(x_list) * w)
                x_max = int(max(x_list) * w)
                y_min = int(min(y_list) * h)
                y_max = int(max(y_list) * h)

                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size == 0:
                    continue

                # ===== PREDICT =====
                input_tensor = preprocess_hand(hand_img)
                preds = model.predict(input_tensor, verbose=0)[0]

                pred_buffer.append(preds)
                avg_preds = np.mean(pred_buffer, axis=0)

                confidence = np.max(avg_preds)
                class_id = np.argmax(avg_preds)
                label = class_names[class_id]

                if confidence >= CONF_THRESHOLD:
                    text = f"{label} ({confidence*100:.1f}%)"
                    color = (0, 255, 0)
                else:
                    text = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(
                    frame,
                    text,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

        # ===== FPS =====
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        cv2.imshow("Sign Language Recognition (IP Webcam)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
