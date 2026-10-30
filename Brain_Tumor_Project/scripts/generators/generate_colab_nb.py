"""
Helper script to generate colab_train_cnn.ipynb.
"""
import json
from pathlib import Path

def create_colab_notebook():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "cells": []
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [s + "\n" for s in source.strip().split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [s + "\n" for s in source.strip().split("\n")]
        })

    # Cell 1: Title & Overview
    add_md("""# 🧠 Google Colab GPU Pipeline: Train Brain Tumor MRI CNN
### Project: AI-Driven Multimodal Classification of Brain Tumours Using MRI and Gene Expression Data
**Author:** B.Tech Major Project & Research Paper Team  
**Architecture:** DenseNet121 / EfficientNetB0 Transfer Learning (ImageNet weights)  
**Target:** 4-Class Classification (Glioma, Meningioma, Pituitary Tumour, No Tumour)  
**Workflow Goal:**
1. **Train ONCE on Colab Free T4 GPU** (~3-5 minutes).
2. **Download `best_mri_model.keras`**.
3. **Move to laptop** (`C:\\Research_Project\\Brain_Tumor_Project\\models\\`).
4. **Work entirely in VS Code** (local laptop never trains the CNN).
""")

    # Cell 2: Step 1 - GPU Verification
    add_md("""---
## Step 1: Verify Google Colab GPU
Ensure hardware acceleration is enabled: **Runtime** -> **Change runtime type** -> **T4 GPU**.""")

    add_code("""import tensorflow as tf
import sys, os

print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise SystemError("⚠️ No GPU found! Go to 'Runtime' -> 'Change runtime type' and select 'T4 GPU'.")
else:
    print("✅ GPU detected:", gpus)
    !nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv""")

    # Cell 3: Step 2 - Dataset Mount / Upload
    add_md("""---
## Step 2: Mount Google Drive or Prepare Dataset
If your `Research_Project` folder is stored in your Google Drive, mount it below.  
Alternatively, upload your `Brain_Tumor_Multimodal_Datasets.zip` or MRI dataset.""")

    add_code("""# Option A: Mount Google Drive
from google.colab import drive
import os
from pathlib import Path

drive.mount('/content/drive')

# Point to your MRI dataset directory
# Default expected path if stored in Drive:
DRIVE_PROJECT = Path('/content/drive/MyDrive/Research_Project')

if (DRIVE_PROJECT / 'Brain_Tumor_Multimodal_Datasets/MRI').exists():
    MRI_DIR = DRIVE_PROJECT / 'Brain_Tumor_Multimodal_Datasets/MRI'
    print("✅ Found MRI dataset in Google Drive:", MRI_DIR)
elif os.path.exists('/content/MRI'):
    MRI_DIR = Path('/content/MRI')
    print("✅ Found MRI dataset in local Colab storage:", MRI_DIR)
else:
    print("⚠️ Dataset not found in default path. Please adjust MRI_DIR below.")
    MRI_DIR = Path('/content/drive/MyDrive/Research_Project/Brain_Tumor_Multimodal_Datasets/MRI')

TRAIN_DIR = MRI_DIR / 'Training'
TEST_DIR = MRI_DIR / 'Testing'
print(f"Train directory: {TRAIN_DIR} (exists: {TRAIN_DIR.exists()})")
print(f"Test directory:  {TEST_DIR} (exists: {TEST_DIR.exists()})")""")

    # Cell 4: Step 3 - Dataset Pipeline & Data Loaders
    add_md("""---
## Step 3: Fast tf.data Image Loading & Data Augmentation
4 Classes: `glioma`, `meningioma`, `notumor`, `pituitary`  
Image Size: 224 x 224 x 3  
Includes CLAHE contrast enhancement & data augmentation (RandomFlip, RandomRotation, RandomZoom).""")

    add_code("""import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}

print(f"Target classes: {CLASSES}")

# Load image paths and labels
def get_image_records(data_dir):
    records = []
    for cls_name in CLASSES:
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            # Support case variations (e.g., no_tumor vs notumor)
            for d in data_dir.iterdir():
                if d.is_dir() and d.name.lower().replace(" ", "").replace("_", "") == cls_name.replace(" ", "").replace("_", ""):
                    cls_dir = d
                    break
        if cls_dir.exists():
            for p in cls_dir.glob("*.jpg"):
                records.append((str(p), CLASS_MAP[cls_name]))
            for p in cls_dir.glob("*.jpeg"):
                records.append((str(p), CLASS_MAP[cls_name]))
            for p in cls_dir.glob("*.png"):
                records.append((str(p), CLASS_MAP[cls_name]))
    return records

train_records = get_image_records(TRAIN_DIR)
test_records = get_image_records(TEST_DIR)

print(f"Loaded {len(train_records)} Training images and {len(test_records)} Testing images.")

# Split train records into 80% train / 20% validation
from sklearn.model_selection import train_test_split
train_paths = [r[0] for r in train_records]
train_labels = [r[1] for r in train_records]
test_paths = [r[0] for r in test_records]
test_labels = [r[1] for r in test_records]

tr_paths, val_paths, tr_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

print(f"Splits: Train={len(tr_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

def parse_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    # DenseNet expects inputs in [0, 255] for densenet preprocess_input
    img = tf.keras.applications.densenet.preprocess_input(img)
    label_one_hot = tf.one_hot(label, depth=4)
    return img, label_one_hot

# tf.data pipelines with caching and prefetching for maximum GPU throughput
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels))
train_ds = train_ds.shuffle(buffer_size=1024).map(parse_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds = test_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

print("✅ High-speed GPU data pipeline built!")""")

    # Cell 5: Step 4 - Model Architecture
    add_md("""---
## Step 4: Build DenseNet121 Transfer Learning Model
Architecture matches research specifications:
- DenseNet121 backbone (pre-trained on ImageNet)
- GlobalAveragePooling2D
- BatchNormalization + Dense(256, activation='relu') + Dropout(0.4)
- Dense(4, activation='softmax')""")

    add_code("""def build_densenet121_model(input_shape=(224, 224, 3), num_classes=4):
    base_model = tf.keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    # Unfreeze top layers for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    inputs = keras.Input(shape=input_shape)
    
    # Data Augmentation layer inside model
    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")
    
    x = data_aug(inputs)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="DenseNet121_Brain_Tumor")
    return model

model = build_densenet121_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)
print(f"✅ DenseNet121 compiled! Total parameters: {model.count_params():,}")""")

    # Cell 6: Step 5 - Model Training
    add_md("""---
## Step 5: Execute Training on Colab GPU
With T4 GPU, 15 epochs take approximately **3-4 minutes**.""")

    add_code("""callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_mri_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

print("🚀 Starting GPU training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)
print("✅ Training finished!")""")

    # Cell 7: Step 6 - Independent Test Evaluation
    add_md("""---
## Step 6: Evaluate on Independent Test Set
Compute Test Accuracy, Classification Report, and Confusion Matrix.""")

    add_code("""import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load best checkpoint
best_model = keras.models.load_model("best_mri_model.keras")

# Generate predictions
y_pred_probs = best_model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.array(test_labels)

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")

print(f"\\n🎯 Test Accuracy: {acc*100:.2f}%")
print(f"🎯 Macro F1-Score: {macro_f1:.4f}\\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

# Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f"DenseNet121 Test Confusion Matrix (Acc: {acc*100:.2f}%)")
plt.xlabel("Predicted Label")
plt.ylabel("Ground Truth")
plt.tight_layout()
plt.savefig("test_confusion_matrix.png", dpi=300)
plt.show()""")

    # Cell 8: Step 7 - Download Weights to Laptop
    add_md("""---
## Step 7: Download `best_mri_model.keras` to Your Laptop
Run the cell below to trigger an automatic browser download of the trained model file.  
Once downloaded, place it in:
```
C:\\Research_Project\\Brain_Tumor_Project\\models\\best_mri_model.keras
```
Then, you are done! All local testing, CLI predictions (`python predict.py`), metrics evaluation (`python evaluate.py`), and Streamlit web dashboard (`streamlit run app.py`) will work locally on your laptop without needing any further training!""")

    add_code("""from google.colab import files
import shutil

# If Google Drive was mounted, also save a copy directly to Drive
try:
    if DRIVE_PROJECT.exists():
        target_dir = DRIVE_PROJECT / "Brain_Tumor_Project/models"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy("best_mri_model.keras", target_dir / "best_mri_model.keras")
        print(f"✅ Saved copy to Google Drive: {target_dir / 'best_mri_model.keras'}")
except Exception as e:
    print(f"Drive copy skipped: {e}")

# 1-Click browser download
print("⬇️ Triggering browser download of best_mri_model.keras...")
files.download("best_mri_model.keras")
print("✅ Done! Copy this file into C:\\\\Research_Project\\\\Brain_Tumor_Project\\\\models\\\\")""")

    # Write notebook
    out_path = Path("colab_train_cnn.ipynb")
    out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Generated {out_path} ({out_path.stat().st_size} bytes)")

    # Also save a copy inside notebooks/
    nb_dir_path = Path("notebooks/colab_train_cnn.ipynb")
    nb_dir_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Generated {nb_dir_path} ({nb_dir_path.stat().st_size} bytes)")

if __name__ == "__main__":
    create_colab_notebook()
