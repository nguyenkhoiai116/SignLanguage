import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os

#cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

safe = 40
imgSize = 300

video = "video"
folder = "data"

for video in os.listdir(video):
    if not video.endswith((".mp4", ".avi", ".MOV", ".mov")):
        continue
    videoPath = os.path.join("video", video)
    ten = os.path.splitext(video)[0]
    print(" xu ly video : ", videoPath)    
    cap = cv2.VideoCapture(videoPath)
    Save = os.path.join(folder, ten)
    os.makedirs(Save, exist_ok=True)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(0.5* fps) # lay 1 frame moi giay
    frame_id = 0 
    while True:
        success, img = cap.read()
        if not success:
            break
        frame_id += 1
        if frame_id % frame_interval != 0:
                continue
        hands, img = detector.findHands(img, draw=False)
        if hands:
            hand = hands[0]
            x,y,w,h = hand['bbox'] # lay vi tri tay
            # print(x,y,w,h)
            h_img, w_img, _ = img.shape
            x1 = max(0, x - safe) 
            y1 = max(0, y - safe)
            x2 = min(w_img, x + w + safe)
            y2 = min(h_img, y + h + safe)
            imgCrop = img[y1:y2, x1:x2] # cat vung tay
            # print(imgCrop.shape)
            if imgCrop.size == 0:
                continue    
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 # tao anh trang de chuan hoa
            hCrop, wCrop, _ = imgCrop.shape
            Ti_Le = hCrop / wCrop
            
            if Ti_Le > 1: # neu chieu cao lon hon chieu rong
                k = imgSize/hCrop
                wCal = int(k*wCrop) # 
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal)//2
                imgWhite[:, wGap:wCal + wGap] = imgResize
            
            else : # neu chieu rong lon hon chieu cao
                k = imgSize/wCrop
                hCal = int(k*hCrop)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal)//2
                imgWhite[hGap:hCal + hGap, :] = imgResize
        
            Save = os.path.join(folder, ten, f"{ten}_{frame_id}.jpg")
            cv2.imwrite(Save, imgWhite)
            print(f"Da luu anh: {Save}")
            
        #     cv2.imshow("Image White", imgWhite) # hien thi anh trang
        #     cv2.imshow("Image Crop", imgCrop) # hien thi vung tay cat duoc
        # cv2.imshow("IMG", img) # hien thi frame
    cap.release()
cv2.destroyAllWindows()

    
