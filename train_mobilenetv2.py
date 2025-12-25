"""
train_mobilenetv2.py
---------------------------------
Huấn luyện mô hình nhận diện Sign Language
- Backbone: MobileNetV2
- Regularization: L1 (ép feature yếu)
- Output: Softmax (hiển thị % confidence)
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

from load_data import load_data

# ======================= CONFIG =======================
IMG_SIZE = 224
EPOCHS_STAGE_1 = 15
EPOCHS_STAGE_2 = 20

LR_STAGE_1 = 1e-3
LR_STAGE_2 = 1e-4

L1_LAMBDA = 1e-4  # Ép feature yếu về 0
MODEL_NAME = "mobilenetv2_sign_language.h5"
# ======================================================


def build_model(num_classes):
    """
    Xây dựng mô hình MobileNetV2 + Head
    """

    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    # ===== Freeze backbone (Stage 1) =====
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=l1(L1_LAMBDA)
    )(x)

    x = Dropout(0.5)(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    return model, base_model


def train():
    # ================= LOAD DATA =================
    train_data, val_data, test_data, NUM_CLASSES, CLASS_NAMES = load_data()

    # ================= BUILD MODEL =================
    model, base_model = build_model(NUM_CLASSES)

    # ================= STAGE 1 =================
    print("\n🚀 STAGE 1: Train classifier head")

    model.compile(
        optimizer=Adam(learning_rate=LR_STAGE_1),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(
            MODEL_NAME,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            verbose=1
        )
    ]

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_STAGE_1,
        callbacks=callbacks
    )

    # ================= STAGE 2 =================
    print("\n🔥 STAGE 2: Fine-tuning backbone")

    # Unfreeze một phần backbone
    for layer in base_model.layers[-40:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=LR_STAGE_2),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_STAGE_2,
        callbacks=callbacks
    )

    # ================= EVALUATION =================
    print("\n📊 Evaluate on test set")
    loss, acc = model.evaluate(test_data)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    # ================= SAVE FINAL MODEL =================
    model.save(MODEL_NAME)
    print(f"\n✅ Model saved as: {MODEL_NAME}")

    # ================= SAVE CLASS NAMES =================
    with open("class_names.txt", "w", encoding="utf-8") as f:
        for name in CLASS_NAMES:
            f.write(name + "\n")

    print("✅ class_names.txt saved")


if __name__ == "__main__":
    train()
