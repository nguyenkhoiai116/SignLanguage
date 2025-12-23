import cv2
import mediapipe as mp
import os
import time

# ================== CONFIG ==================
DATASET_DIR = 'data'
IMG_SIZE = 224
DELAY = 0.15
BOX_MARGIN = 1.4
IP_CAMERA_URL = 'http://10.0.0.67:8080/video'
# ============================================

# ===== NHẬP TÊN FOLDER =====
LABEL = input("👉 Nhập tên folder để lưu ảnh (VD: A, B, CH, DAU_NGA): ").strip()

if LABEL == '':
    print("❌ Tên folder không hợp lệ")
    exit()

save_path = os.path.join(DATASET_DIR, LABEL)
os.makedirs(save_path, exist_ok=True)

# ===== LẤY CHỈ SỐ ẢNH TIẾP THEO =====
def get_next_index(folder, label):
    max_idx = 0
    for f in os.listdir(folder):
        if f.startswith(label + "_") and f.endswith(".jpg"):
            try:
                idx = int(f.split("_")[-1].split(".")[0])
                max_idx = max(max_idx, idx)
            except:
                pass
    return max_idx + 1

count = get_next_index(save_path, LABEL)

print(f"📁 Lưu ảnh vào: {save_path}")
print(f"📸 Bắt đầu từ: {LABEL}_{str(count).zfill(3)}.jpg")
print("👉 Nhấn ESC để dừng")

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===== CAMERA =====
cap = cv2.VideoCapture(IP_CAMERA_URL)
if not cap.isOpened():
    print("❌ Không kết nối được camera")
    exit()

last_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Mất tín hiệu camera")
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        xs = [lm.x * w for lm in hand.landmark]
        ys = [lm.y * h for lm in hand.landmark]

        xmin, xmax = int(min(xs)), int(max(xs))
        ymin, ymax = int(min(ys)), int(max(ys))

        box_size = int(max(xmax - xmin, ymax - ymin) * BOX_MARGIN)
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2

        x1 = max(cx - box_size // 2, 0)
        y1 = max(cy - box_size // 2, 0)
        x2 = min(cx + box_size // 2, w)
        y2 = min(cy + box_size // 2, h)

        hand_crop = frame[y1:y2, x1:x2]

        if hand_crop.size != 0:
            hand_crop = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)

            now = time.time()
            if now - last_time > DELAY:
                filename = f"{LABEL}_{str(count).zfill(3)}.jpg"
                cv2.imwrite(os.path.join(save_path, filename), gray)
                count += 1
                last_time = now

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f'{LABEL} | {count-1} images',
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow('Collect Sign Language Data', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Hoàn tất: {count-1} ảnh cho lớp '{LABEL}'")
