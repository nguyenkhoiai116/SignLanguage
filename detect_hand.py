import cv2
import mediapipe as mp
import os

# =======================
# CONFIG
# =======================

VIDEO_DIR = "split video"          # Thư mục chứa nhiều video
OUTPUT_DIR = "raw"   # Thư mục lưu ảnh tay
IMG_SIZE = 224
SAVE_EVERY_N_FRAME = 20            # Lưu mỗi N frame

os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# =======================
# KHỞI TẠO MEDIAPIPE HANDS
# =======================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =======================
# DUYỆT QUA TỪNG VIDEO
# =======================

save_id = 0  # global counter

for video_name in os.listdir(VIDEO_DIR):

    if not video_name.lower().endswith(VIDEO_EXTS):
        continue

    video_path = os.path.join(VIDEO_DIR, video_name)
    print(f"\n▶ Đang xử lý video: {video_name}")

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id % SAVE_EVERY_N_FRAME != 0:
            continue

        h, w, _ = frame.shape

        # =======================
        # MEDIAPIPE PROCESS
        # =======================

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # -----------------------
                # BOUNDING BOX
                # -----------------------

                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]

                x1 = int(min(xs) * w)
                y1 = int(min(ys) * h)
                x2 = int(max(xs) * w)
                y2 = int(max(ys) * h)

                pad = 100
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                # -----------------------
                # CROP + RESIZE
                # -----------------------

                hand_crop = frame[y1:y2, x1:x2]
                if hand_crop.size == 0:
                    continue

                hand_crop_resized = cv2.resize(
                    hand_crop, (IMG_SIZE, IMG_SIZE)
                )

                # -----------------------
                # PREVIEW
                # -----------------------

                cv2.imshow("Frame (BBox Preview)", frame)
                cv2.imshow("Hand Preview", hand_crop_resized)

                # -----------------------
                # LƯU ẢNH (gắn tên video)
                # -----------------------

                video_id = os.path.splitext(video_name)[0]

                cv2.imwrite(
                    f"{OUTPUT_DIR}/{video_id}_hand_{save_id:06d}.jpg",
                    hand_crop_resized
                )

                save_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            hands.close()
            cv2.destroyAllWindows()
            print("⛔ Dừng bởi người dùng")
            exit()

    cap.release()

# =======================
# CLEAN UP
# =======================

hands.close()
cv2.destroyAllWindows()
print("\nDONE! Tổng số ảnh đã lưu:", save_id)
