import os
from collections import Counter

def count_images_per_class(root_dir):
    counter = Counter()

    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        num_images = len([
            f for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        counter[cls] = num_images

    return counter


if __name__ == "__main__":
    train_dir = "dataset/train"
    val_dir   = "dataset/val"

    print("📊 TRAIN SET DISTRIBUTION")
    train_count = count_images_per_class(train_dir)
    for cls, n in train_count.items():
        print(f"{cls}: {n}")

    print("\n📊 VAL SET DISTRIBUTION")
    val_count = count_images_per_class(val_dir)
    for cls, n in val_count.items():
        print(f"{cls}: {n}")
