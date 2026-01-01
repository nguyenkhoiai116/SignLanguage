import os

DATA_DIR = "data\\clean"
IMG_EXTS = (".jpg", ".jpeg", ".png")

for folder_name in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder_name)

    # Bỏ qua nếu không phải folder
    if not os.path.isdir(folder_path):
        continue

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(IMG_EXTS)
    ]

    images.sort()

    for idx, img in enumerate(images, start=1):
        old_path = os.path.join(folder_path, img)

        new_name = f"{folder_name}_{str(idx).zfill(3)}.jpg"
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

    print(f"✅ Folder '{folder_name}': đổi tên {len(images)} ảnh")

print("🎉 Hoàn tất (folder giữ nguyên)")