import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import mediapipe as mp
import numpy as np
import os


MODEL_PATH = "best_model_checkpoint.pth"
CLASS_NAMES = sorted(os.listdir("dataset/train"))  # lấy theo folder
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.5   # 50%
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, len(CLASS_NAMES))
)

def preprocess_hand_like_train(imgCrop, imgSize=224):
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    hCrop, wCrop, _ = imgCrop.shape
    ratio = hCrop / wCrop

    if ratio > 1:
        k = imgSize / hCrop
        wCal = int(k * wCrop)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = (imgSize - wCal) // 2
        imgWhite[:, wGap:wCal + wGap] = imgResize
    else:
        k = imgSize / wCrop
        hCal = int(k * hCrop)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = (imgSize - hCal) // 2
        imgWhite[hGap:hCal + hGap, :] = imgResize

    return imgWhite

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
cap = cv2.VideoCapture(0)

print("🎥 Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]

            x1 = int(min(xs) * w)
            x2 = int(max(xs) * w)
            y1 = int(min(ys) * h)
            y2 = int(max(ys) * h)
            margin = 40
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            hand_crop = frame[y1:y2, x1:x2]
            if hand_crop.size == 0:
                continue
            hand_img = preprocess_hand_like_train(hand_crop, IMG_SIZE)
            input_tensor = transform(hand_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            confidence = conf.item()
            if confidence >= CONF_THRESHOLD:
                label = CLASS_NAMES[pred.item()]
                text = f"{label} ({confidence*100:.1f}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "Unknown",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

