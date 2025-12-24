import cv2
import mediapipe as mp
import os

# =======================
# CONFIG
# =======================

VIDEO_PATH = "D:\\Document\\HK1_II_2\\NhapMon\\Project\\SignLanguge\\split video\\7350425037408.mp4"     # Đường dẫn video đầu vào
OUTPUT_DIR = "hand_images"   # Thư mục lưu ảnh tay
IMG_SIZE = 224               # Kích thước ảnh đầu ra (224x224)
SAVE_EVERY_N_FRAME = 20       # Lưu mỗi N frame (1 = frame nào cũng lưu)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =======================
# KHỞI TẠO MEDIAPIPE HANDS
# =======================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,     # False = dùng tracking (nhanh hơn video)
    max_num_hands=1,             # Giới hạn 1 tay (đỡ nhiễu)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =======================
# ĐỌC VIDEO
# =======================

cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0   # Đếm số frame
save_id = 0    # Đếm số ảnh đã lưu

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # Bỏ qua frame nếu không đúng chu kỳ lưu
    if frame_id % SAVE_EVERY_N_FRAME != 0:
        continue

    h, w, _ = frame.shape

    # =======================
    # XỬ LÝ MEDIAPIPE
    # =======================

    # OpenCV dùng BGR, MediaPipe cần RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect tay
    results = hands.process(rgb)

    # Nếu phát hiện được tay
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # =======================
            # LẤY BOUNDING BOX TỪ LANDMARK
            # =======================

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            print(hand_landmarks.landmark)

            # Chuyển từ tọa độ chuẩn hóa (0–1) sang pixel
            x1 = int(min(xs) * w) # cạnh trái
            y1 = int(min(ys) * h) # cạnh trên   
            x2 = int(max(xs) * w) # cạnh phải
            y2 = int(max(ys) * h) # cạnh dưới

            # Nới bbox để tránh cụt tay
            pad = 40 # pixels
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            # =======================
            # VẼ PREVIEW BBOX
            # =======================

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            # =======================
            # CROP ẢNH BÀN TAY
            # =======================

            hand_crop = frame[y1:y2, x1:x2]

            if hand_crop.size == 0:
                continue

            # Resize về kích thước cố định
            hand_crop_resized = cv2.resize(
                hand_crop, (IMG_SIZE, IMG_SIZE)
            )

            # =======================
            # PREVIEW TRƯỚC KHI LƯU
            # =======================

            cv2.imshow("Frame (BBox Preview)", frame)
            cv2.imshow("Hand Preview (224x224)", hand_crop_resized)

            # =======================
            # LƯU ẢNH
            # =======================

            cv2.imwrite(
                f"{OUTPUT_DIR}/hand_{save_id:05d}.jpg",
                hand_crop_resized
            )
            save_id += 1

    # =======================
    # THOÁT BẰNG PHÍM Q
    # =======================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =======================
# GIẢI PHÓNG TÀI NGUYÊN
# =======================

cap.release()
hands.close()
cv2.destroyAllWindows()

print("Done! Tổng số ảnh đã lưu:", save_id)
