import os
import shutil
import random
import tensorflow as tf

SRC_DIR = "data\clean"
DST_DIR = "dataset"

TRAIN = 0.7 
VAL = 0.15
TEST = 0.15
random.seed(42) 

for split in ["TRAIN", "val", "test"]:
    os.makedirs(os.path.join(DST_DIR, split), exist_ok=True)
for class_name in os.listdir(SRC_DIR):
    class_path = os.path.join(SRC_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    images = os.listdir(class_path)
    random.shuffle(images)
    n_total = len(images)
    n_TRAIN = int(n_total * TRAIN)
    n_val = int(n_total * VAL)
    splits = {
        "train": images[:n_TRAIN],
        "val": images[n_TRAIN:n_TRAIN + n_val],
        "test": images[n_TRAIN + n_val:]
    }

    for split, split_imgs in splits.items():
        split_class_dir = os.path.join(DST_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for img in split_imgs:
            src_img = os.path.join(class_path, img)
            dst_img = os.path.join(split_class_dir, img)
            shutil.copy(src_img, dst_img)

    print(f"✅ {class_name}: {n_total} images "
          f"(train={len(splits['train'])}, "
          f"val={len(splits['val'])}, "
          f"test={len(splits['test'])})")
