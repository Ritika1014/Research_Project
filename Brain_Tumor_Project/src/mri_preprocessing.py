"""
MRI Preprocessing Module
Provides reusable image processing operations for brain MRI datasets:
- Resizing
- Grayscale / RGB conversion
- Pixel normalization
- Contrast enhancement (CLAHE, Histogram Equalization)
- Denoising (Gaussian Blur, Median Blur)
- Intensity adjustments
- Batch dataset processing
"""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    imagehash = None
    HAS_IMAGEHASH = False
import pandas as pd


def load_image(image_path):
    """
    Load an image from file path using OpenCV.
    Returns RGB numpy array or None if loading fails.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def resize_image(img, size=(224, 224)):
    """
    Resize image to target size (width, height).
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def convert_to_rgb(img):
    """
    Ensure image has 3 color channels (RGB).
    """
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def convert_to_grayscale(img):
    """
    Convert RGB image to 1-channel Grayscale.
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def normalize_pixels(img, range_type='0_1'):
    """
    Normalize pixel values to [0, 1] float32 or [0, 255] uint8.
    """
    if range_type == '0_1':
        return img.astype(np.float32) / 255.0
    elif range_type == '0_255':
        norm = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
        return norm.astype(np.uint8)
    return img


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Works on grayscale or RGB (LAB color space).
    """
    if len(img.shape) == 3:
        # Convert RGB to LAB, apply CLAHE on L channel, then back to RGB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)


def apply_histogram_equalization(img):
    """
    Apply standard global histogram equalization.
    """
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        eq_img = cv2.merge((y_eq, cr, cb))
        return cv2.cvtColor(eq_img, cv2.COLOR_YCrCb2RGB)
    else:
        return cv2.equalizeHist(img)


def apply_gaussian_blur(img, kernel_size=(5, 5), sigma=1.0):
    """
    Apply Gaussian Smoothing Blur.
    """
    return cv2.GaussianBlur(img, kernel_size, sigma)


def apply_median_blur(img, kernel_size=5):
    """
    Apply Median Blur filter for noise reduction.
    """
    return cv2.medianBlur(img, kernel_size)


def adjust_brightness_contrast(img, alpha=1.2, beta=10):
    """
    Adjust brightness (beta) and contrast (alpha).
    new_img = alpha * img + beta
    """
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def preprocess_pipeline(img, target_size=(224, 224), use_clahe=True, 
                        use_blur=True, blur_type='gaussian', normalize=False):
    """
    Full sequential preprocessing pipeline for a single MRI image:
    1. Ensure RGB format
    2. Optional Denoising (Gaussian or Median blur)
    3. Optional Contrast Enhancement (CLAHE)
    4. Resize to target dimensions
    5. Optional Normalization to [0, 1]
    """
    processed = convert_to_rgb(img)
    
    if use_blur:
        if blur_type == 'gaussian':
            processed = apply_gaussian_blur(processed, kernel_size=(3, 3), sigma=0.5)
        elif blur_type == 'median':
            processed = apply_median_blur(processed, kernel_size=3)
            
    if use_clahe:
        processed = apply_clahe(processed, clip_limit=2.0, tile_grid_size=(8, 8))
        
    processed = resize_image(processed, target_size)
    
    if normalize:
        processed = normalize_pixels(processed, range_type='0_1')
        
    return processed


def compute_image_hash(image_path, hash_func=None):
    """
    Compute perceptual hash for duplicate detection.
    """
    if not HAS_IMAGEHASH and hash_func is None:
        return None
    try:
        fn = hash_func or imagehash.phash
        with Image.open(image_path) as pil_img:
            return str(fn(pil_img))
    except Exception:
        return None


def scan_mri_dataset(data_dir):
    """
    Scan an MRI dataset folder, return detailed summary dataframe and stats.
    Counts files, checks image corruption, records sizes and channels.
    """
    records = []
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for cls_name in classes:
        cls_folder = os.path.join(data_dir, cls_name)
        for fname in os.listdir(cls_folder):
            fpath = os.path.join(cls_folder, fname)
            if not os.path.isfile(fpath):
                continue
                
            file_size_bytes = os.path.getsize(fpath)
            
            # Read image metadata
            try:
                with Image.open(fpath) as pil_img:
                    width, height = pil_img.size
                    mode = pil_img.mode
                    is_corrupted = False
            except Exception:
                width, height = 0, 0
                mode = 'UNKNOWN'
                is_corrupted = True
                
            records.append({
                'filename': fname,
                'path': fpath,
                'class': cls_name,
                'file_size_bytes': file_size_bytes,
                'width': width,
                'height': height,
                'aspect_ratio': round(width / height, 2) if height > 0 else 0,
                'mode': mode,
                'is_corrupted': is_corrupted
            })
            
    return pd.DataFrame(records)


def find_duplicate_images(df, max_sample=1000):
    """
    Find duplicate images using perceptual hashing across dataset.
    Samples up to max_sample images for ultra-fast duplicate detection.
    """
    hash_dict = {}
    sampled_df = df[~df['is_corrupted']]
    if len(sampled_df) > max_sample:
        sampled_df = sampled_df.sample(n=max_sample, random_state=42)
        
    for idx, row in sampled_df.iterrows():
        phash = compute_image_hash(row['path'])
        if phash:
            if phash not in hash_dict:
                hash_dict[phash] = []
            hash_dict[phash].append(row['path'])
                    
    duplicates = {h: paths for h, paths in hash_dict.items() if len(paths) > 1}
    return duplicates


def batch_process_mri_dataset(src_dir, dst_dir, target_size=(224, 224), use_clahe=True, use_blur=True):
    """
    Batch process all MRI images from src_dir to dst_dir maintaining class subfolder structure.
    Returns metadata DataFrame of processed images.
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    records = []
    
    for cls_dir in src_path.iterdir():
        if not cls_dir.is_dir():
            continue
            
        out_cls_dir = dst_path / cls_dir.name
        out_cls_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in cls_dir.glob('*.*'):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                continue
                
            raw_img = load_image(img_file)
            if raw_img is None:
                continue
                
            orig_h, orig_w = raw_img.shape[:2]
            proc_img = preprocess_pipeline(raw_img, target_size=target_size, use_clahe=use_clahe, use_blur=use_blur, normalize=False)
            
            save_path = out_cls_dir / f"{img_file.stem}.jpg"
            # Save as RGB JPEG
            cv2.imwrite(str(save_path), cv2.cvtColor(proc_img, cv2.COLOR_RGB2BGR))
            
            records.append({
                'original_filename': img_file.name,
                'class': cls_dir.name,
                'original_path': str(img_file),
                'processed_path': str(save_path),
                'original_width': orig_w,
                'original_height': orig_h,
                'processed_width': target_size[0],
                'processed_height': target_size[1],
                'channels': 3
            })
            
    return pd.DataFrame(records)
