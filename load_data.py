"""
load_data.py
---------------------------------
Nhiệm vụ:
- Load dataset đã chia (train / val / test)
- Resize ảnh đúng chuẩn MobileNetV2 (224x224)
- Normalize ảnh (01)
- Data augmentation cho train
- Tạo mapping class index → tên ký hiệu
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======================= CONFIG =======================
IMG_SIZE = 224
BATCH_SIZE = 32

TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"
TEST_DIR  = "dataset/test"
# ======================================================


def load_data():
    """
    Trả về:
    - train_data
    - val_data
    - test_data
    - NUM_CLASSES
    - CLASS_NAMES
    """

    # ================= TRAIN =================
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    # ================= VALIDATION =================
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    val_data = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    # ================= TEST =================
    test_gen = ImageDataGenerator(rescale=1.0 / 255)

    test_data = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    # ================= CLASS INFO =================
    NUM_CLASSES = train_data.num_classes
    CLASS_NAMES = list(train_data.class_indices.keys())

    print("\n📊 DATASET INFO")
    print("Số classes:", NUM_CLASSES)
    print("Tên classes:", CLASS_NAMES)
    print("Số ảnh train:", train_data.samples)
    print("Số ảnh val  :", val_data.samples)
    print("Số ảnh test :", test_data.samples)

    return train_data, val_data, test_data, NUM_CLASSES, CLASS_NAMES


# ======================= TEST FILE =======================
if __name__ == "__main__":
    load_data()
