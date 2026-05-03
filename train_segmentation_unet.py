"""
Train U-Net for tumor segmentation boundary detection.
Generates pseudo ground-truth masks using fast CV pipeline
(skull-strip + K-Means + blob scoring — NO GrabCut for speed).
The trained U-Net then produces much better boundaries than heuristics.
"""

import os
import sys
import glob
import time
import numpy as np
import cv2

# Force unbuffered output
import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU to avoid GPU conflict with app.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ── Config ──────────────────────────────────────────────────────────
DATASET_ROOT = os.path.join("dataset", "brain")
TUMOR_FOLDERS = ["glioma", "meningioma", "pituitary"]
MODEL_SAVE_PATH = os.path.join("models", "tumor_segmentation_unet.h5")
IMG_SIZE = 128
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15


# ====================================================================
#  FAST pseudo ground-truth mask generator (no GrabCut)
# ====================================================================

def generate_pseudo_mask(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    from models.segmentation_model import TumorSegmentationModel
    model = TumorSegmentationModel(model_path="dummy")
    
    norm_img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mask = model._heuristic_mask(norm_img)
    
    if np.count_nonzero(mask) < 30:
        return None
        
    img_r = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    mask_r = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
    return img_r, mask_r


# ====================================================================
#  U-Net (3-level encoder-decoder with skip connections)
# ====================================================================

def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder 1
    d1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Conv2D(32, 3, activation="relu", padding="same")(d1)
    d1 = layers.BatchNormalization()(d1)
    p1 = layers.MaxPooling2D(2)(d1)

    # Encoder 2
    d2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Conv2D(64, 3, activation="relu", padding="same")(d2)
    d2 = layers.BatchNormalization()(d2)
    p2 = layers.MaxPooling2D(2)(d2)

    # Encoder 3
    d3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Conv2D(128, 3, activation="relu", padding="same")(d3)
    d3 = layers.BatchNormalization()(d3)
    p3 = layers.MaxPooling2D(2)(d3)

    # Bridge
    b = layers.Conv2D(256, 3, activation="relu", padding="same")(p3)
    b = layers.BatchNormalization()(b)
    b = layers.Conv2D(256, 3, activation="relu", padding="same")(b)
    b = layers.BatchNormalization()(b)
    b = layers.Dropout(0.3)(b)

    # Decoder 3
    u3 = layers.UpSampling2D(2)(b)
    u3 = layers.Conv2D(128, 2, activation="relu", padding="same")(u3)
    u3 = layers.Concatenate()([u3, d3])
    u3 = layers.Conv2D(128, 3, activation="relu", padding="same")(u3)
    u3 = layers.BatchNormalization()(u3)
    u3 = layers.Conv2D(128, 3, activation="relu", padding="same")(u3)
    u3 = layers.BatchNormalization()(u3)

    # Decoder 2
    u2 = layers.UpSampling2D(2)(u3)
    u2 = layers.Conv2D(64, 2, activation="relu", padding="same")(u2)
    u2 = layers.Concatenate()([u2, d2])
    u2 = layers.Conv2D(64, 3, activation="relu", padding="same")(u2)
    u2 = layers.BatchNormalization()(u2)
    u2 = layers.Conv2D(64, 3, activation="relu", padding="same")(u2)
    u2 = layers.BatchNormalization()(u2)

    # Decoder 1
    u1 = layers.UpSampling2D(2)(u2)
    u1 = layers.Conv2D(32, 2, activation="relu", padding="same")(u1)
    u1 = layers.Concatenate()([u1, d1])
    u1 = layers.Conv2D(32, 3, activation="relu", padding="same")(u1)
    u1 = layers.BatchNormalization()(u1)
    u1 = layers.Conv2D(32, 3, activation="relu", padding="same")(u1)
    u1 = layers.BatchNormalization()(u1)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="tumor_mask")(u1)
    return keras.Model(inputs, outputs, name="tumor_unet")


# ====================================================================
#  Loss / Metrics
# ====================================================================

def dice_coeff(y_true, y_pred, smooth=1.0):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coeff(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# ====================================================================
#  Main
# ====================================================================

def main():
    print("=" * 60)
    print("  U-Net Tumor Segmentation Training")
    print("=" * 60)

    # 1. Collect image paths (tumor classes only)
    print("\n[1/4] Collecting tumor image paths ...")
    paths = []
    for split in ["Training", "Testing"]:
        for folder in TUMOR_FOLDERS:
            found = glob.glob(os.path.join(DATASET_ROOT, split, folder, "*.jpg"))
            paths.extend(found)
            print(f"  {split}/{folder}: {len(found)} images")
    print(f"  Total: {len(paths)} images")

    # 2. Generate pseudo-masks (FAST — no GrabCut)
    print("\n[2/4] Generating pseudo ground-truth masks ...")
    images, masks = [], []
    skipped = 0
    t0 = time.time()
    for idx, path in enumerate(paths):
        if (idx + 1) % 500 == 0 or idx == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 0.01)
            remaining = (len(paths) - idx - 1) / max(rate, 0.01)
            print(f"  [{idx+1}/{len(paths)}] {rate:.1f} img/s, ~{remaining:.0f}s remaining")
        result = generate_pseudo_mask(path)
        if result is None:
            skipped += 1
            continue
        images.append(result[0])
        masks.append(result[1])

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s — {len(images)} usable samples, {skipped} skipped")

    X = np.array(images, dtype=np.float32)[..., np.newaxis]
    Y = np.array(masks, dtype=np.float32)[..., np.newaxis]
    del images, masks  # free memory

    # 3. Shuffle + split
    print("\n[3/4] Building U-Net model ...")
    idx = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]

    val_n = int(len(X) * VALIDATION_SPLIT)
    X_train, X_val = X[:-val_n], X[-val_n:]
    Y_train, Y_val = Y[:-val_n], Y[-val_n:]
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    # 4. Build + compile
    model = build_unet()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=["accuracy", dice_coeff],
    )
    param_count = model.count_params()
    print(f"  U-Net parameters: {param_count:,}")

    # 5. Train
    print(f"\n[4/4] Training for {EPOCHS} epochs ...")
    cb = [
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, save_best_only=True,
            monitor="val_dice_coeff", mode="max", verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_dice_coeff", mode="max",
            factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_dice_coeff", mode="max",
            patience=7, restore_best_weights=True, verbose=1
        ),
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
    )

    # 6. Save
    model.save(MODEL_SAVE_PATH)
    print(f"\n{'=' * 60}")
    print(f"  Model saved to: {MODEL_SAVE_PATH}")
    print(f"{'=' * 60}")

    # 7. Final eval
    val_loss, val_acc, val_dice = model.evaluate(X_val, Y_val, verbose=0)
    print(f"\n  Final Validation Loss : {val_loss:.4f}")
    print(f"  Final Validation Acc  : {val_acc:.4f}")
    print(f"  Final Validation Dice : {val_dice:.4f}")


if __name__ == "__main__":
    main()
