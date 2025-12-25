import os
import shutil
import random

# ======================= CONFIG =======================
RAW_DATASET_DIR = "data_clean"   # data gốc
OUTPUT_DATASET_DIR = "dataset"    # data sau khi chia

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

RANDOM_SEED = 42

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
# ======================================================


def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


def remove_old_dataset(path):
    """Xóa dataset cũ để tránh trộn data"""
    if os.path.exists(path):
        print(f"⚠️ Xóa dataset cũ: {path}")
        shutil.rmtree(path)


def main():
    random.seed(RANDOM_SEED)

    # 1️⃣ Kiểm tra dataset gốc
    if not os.path.exists(RAW_DATASET_DIR):
        raise FileNotFoundError(
            f"❌ Không tìm thấy thư mục: {RAW_DATASET_DIR}"
        )

    # 2️⃣ Xóa dataset cũ nếu có
    remove_old_dataset(OUTPUT_DATASET_DIR)

    class_names = [
        d for d in os.listdir(RAW_DATASET_DIR)
        if os.path.isdir(os.path.join(RAW_DATASET_DIR, d))
    ]

    print(f"🔎 Phát hiện {len(class_names)} classes")

    # 3️⃣ Chia data cho từng class
    for class_name in sorted(class_names):
        class_path = os.path.join(RAW_DATASET_DIR, class_name)

        images = [
            f for f in os.listdir(class_path)
            if is_image_file(f)
        ]

        if len(images) < 10:
            print(f"⚠️ Class '{class_name}' quá ít ảnh ({len(images)}), bỏ qua")
            continue

        random.shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end   = train_end + int(total * VAL_RATIO)

        split_dict = {
            "train": images[:train_end],
            "val":   images[train_end:val_end],
            "test":  images[val_end:]
        }

        print(f"\n📂 Class: {class_name}")
        print(f"  Tổng ảnh : {total}")
        print(f"  Train    : {len(split_dict['train'])}")
        print(f"  Val      : {len(split_dict['val'])}")
        print(f"  Test     : {len(split_dict['test'])}")

        # 4️⃣ Copy ảnh sang thư mục mới
        for split_name, file_list in split_dict.items():
            split_dir = os.path.join(
                OUTPUT_DATASET_DIR,
                split_name,
                class_name
            )
            os.makedirs(split_dir, exist_ok=True)

            for filename in file_list:
                src = os.path.join(class_path, filename)
                dst = os.path.join(split_dir, filename)
                shutil.copy2(src, dst)

    print("\n✅ Chia dataset hoàn tất – SẴN SÀNG TRAIN MODEL")


if __name__ == "__main__":
    main()
