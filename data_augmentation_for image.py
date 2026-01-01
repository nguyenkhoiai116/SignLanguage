import cv2
import numpy as np
import os
import random

# =========================
# AUGMENT 1 ẢNH
# =========================
def augment_image(image):
    rows, cols, _ = image.shape

    # 1. Random rotation
    angle = random.choice([90, 180, 270])
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # 2. Random scaling
    scale = random.uniform(0.8, 1.2)
    scaled = cv2.resize(rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Resize/pad về size gốc
    scaled = cv2.resize(scaled, (cols, rows))

    # 3. Random translation
    tx, ty = random.randint(-20, 20), random.randint(-20, 20)
    M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(scaled, M_shift, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # 4. Random crop
    crop_size = random.uniform(0.7, 1.0)
    crop_w = int(cols * crop_size)
    crop_h = int(rows * crop_size)

    x = random.randint(0, cols - crop_w)
    y = random.randint(0, rows - crop_h)

    cropped = shifted[y:y+crop_h, x:x+crop_w]
    cropped = cv2.resize(cropped, (cols, rows))

    # 5. Brightness & contrast
    brightness = random.randint(-30, 30)
    contrast = random.uniform(0.8, 1.2)
    bc = cv2.convertScaleAbs(cropped, alpha=contrast, beta=brightness)

    # 6. Color shift
    shift = np.array([
        random.randint(-20, 20),
        random.randint(-20, 20),
        random.randint(-20, 20)
    ], dtype=np.int16)

    bc = np.clip(bc.astype(np.int16) + shift, 0, 255).astype(np.uint8)

    return bc


# =========================
# AUGMENT DATASET NHIỀU LABEL
# =========================
def augment_multilabel_dataset(input_root, output_root, num_augmentations=5):
    for label in os.listdir(input_root):
        label_in = os.path.join(input_root, label)

        if not os.path.isdir(label_in):
            continue

        label_out = os.path.join(output_root, label)
        os.makedirs(label_out, exist_ok=True)

        for file in os.listdir(label_in):
            if not file.lower().endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(label_in, file)
            image = cv2.imread(img_path)

            if image is None:
                print(f"❌ Không đọc được {img_path}")
                continue

            # Lưu ảnh gốc
            cv2.imwrite(os.path.join(label_out, file), image)

            name, ext = os.path.splitext(file)

            for i in range(num_augmentations):
                aug = augment_image(image)
                new_name = f"{name}_aug_{i}{ext}"
                cv2.imwrite(os.path.join(label_out, new_name), aug)

        print(f"✅ Label {label} xong")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    input_root = "data"
    output_root = "data_clean"

    augment_multilabel_dataset(
        input_root,
        output_root,
        num_augmentations=9
    )

    print("🎉 Augment xong toàn bộ dataset")
