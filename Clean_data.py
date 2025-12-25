
import cv2
import numpy as np
import os
import random

# =========================
# AUGMENT 1 ẢNH (KHÔNG FLIP)
# =========================
def augment_image(image):
    # Get image dimensions
    height, width, channels = image.shape

    # 1. SHEAR NHẸ
    shearFactor = random.uniform(-0.15, 0.15)
    shearMatrix = np.array([
        [1, shearFactor, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    augmentedImg = cv2.warpAffine(
        image, shearMatrix, (width, height),
        borderMode=cv2.BORDER_REFLECT
    )

    # 2. SHIFT NHẸ
    maxShift = 0.1
    translateX = random.uniform(-maxShift, maxShift) * width
    translateY = random.uniform(-maxShift, maxShift) * height

    shiftMatrix = np.float32([
        [1, 0, translateX],
        [0, 1, translateY]
    ])

    augmentedImg = cv2.warpAffine(
        augmentedImg, shiftMatrix, (width, height),
        borderMode=cv2.BORDER_REFLECT
    )

    # 3. BRIGHTNESS + CONTRAST
    brightnessOffset = random.randint(-40, 40)
    contrastFactor = random.uniform(0.8, 1.2)

    augmentedImg = cv2.convertScaleAbs(
        augmentedImg,
        alpha=contrastFactor,
        beta=brightnessOffset
    )

    # 4. RANDOM GRAYSCALE (50%)
    if random.random() < 0.5:
        grayImg = cv2.cvtColor(augmentedImg, cv2.COLOR_BGR2GRAY)
        augmentedImg = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR)

    return augmentedImg

def augment_multilabel_dataset(
    inputRoot,
    outputRoot,
    numAugmentations=3
):
    # Loop through each label directory
    for labelName in os.listdir(inputRoot):
        labelInputPath = os.path.join(inputRoot, labelName)

        if not os.path.isdir(labelInputPath):
            continue

        labelOutputPath = os.path.join(outputRoot, labelName)
        os.makedirs(labelOutputPath, exist_ok=True)

        # Process each image file in the label directory
        for fileName in os.listdir(labelInputPath):
            if not fileName.lower().endswith((".jpg", ".png")):
                continue

            imgPath = os.path.join(labelInputPath, fileName)
            image = cv2.imread(imgPath)

            if image is None:
                print(f"❌ Không đọc được: {imgPath}")
                continue

            # Lưu ảnh gốc
            cv2.imwrite(
                os.path.join(labelOutputPath, fileName),
                image
            )

            baseName, extension = os.path.splitext(fileName)

            # Sinh ảnh augment
            for i in range(numAugmentations):
                augImg = augment_image(image)
                newFileName = f"{baseName}_aug_{i}{extension}"
                cv2.imwrite(
                    os.path.join(labelOutputPath, newFileName),
                    augImg
                )

        print(f"✅ Label {labelName} xong")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Define input and output directories
    inputRoot = "data"
    outputRoot = "data_clean"

    # Call the augmentation function
    augment_multilabel_dataset(
        inputRoot=inputRoot,
        outputRoot=outputRoot,
        numAugmentations=5
    )

    print("🎉 Augmentation toàn bộ dataset hoàn tất!")
