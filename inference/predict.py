#!/usr/bin/env python3
"""
Sequence OCR prediction script using CRNN + CTC model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMG_WIDTH = 512
IMG_HEIGHT = 64

def preprocess_image(image_path: str) -> np.ndarray:
    """Load a line image, normalize ink to white on black, crop, and pad to model input."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    denoised = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary_inv = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # If the source is already light text on dark background, Otsu+INV selects the background.
    ink_ratio = float(np.count_nonzero(binary_inv)) / binary_inv.size
    if ink_ratio > 0.65:
        _, binary_inv = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(binary_inv)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad_x = max(4, int(w * 0.04))
        pad_y = max(4, int(h * 0.20))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(binary_inv.shape[1], x + w + pad_x)
        y1 = min(binary_inv.shape[0], y + h + pad_y)
        binary = binary_inv[y0:y1, x0:x1]
    else:
        binary = binary_inv

    h, w = binary.shape
    scale = IMG_HEIGHT / h
    new_w = max(1, int(w * scale))

    if new_w > IMG_WIDTH:
        resized = cv2.resize(binary, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    else:
        resized_h = cv2.resize(binary, (new_w, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        padded = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        x_offset = (IMG_WIDTH - new_w) // 2
        padded[:, x_offset:x_offset + new_w] = resized_h
        resized = padded

    # Normalize to [0, 1] and add batch/channel dims
    input_data = resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=(0, -1)) # [1, 64, 512, 1]

    return input_data

def decode_prediction(preds: np.ndarray, labels: list[str], label_to_char: dict) -> tuple[str, list[str]]:
    """Greedy decode predictions from the TFLite output probability matrix."""
    # preds has shape [1, time_steps, NumClasses + 1]
    time_steps = preds.shape[1]
    
    best_path = []
    # Find argmax for each time step
    for t in range(time_steps):
        best_idx = np.argmax(preds[0, t])
        best_path.append(best_idx)
        
    # Apply CTC decoding: collapse repetitions and remove blanks (blank is at the last index)
    blank_idx = len(labels)
    collapsed = []
    prev = -1
    for idx in best_path:
        if idx != prev:
            if idx != blank_idx:
                collapsed.append(idx)
            prev = idx
            
    # Map back to labels and Unicode characters
    decoded_labels = [labels[idx] for idx in collapsed if idx < len(labels)]
    unicode_chars = [label_to_char.get(lbl, lbl) for lbl in decoded_labels]
    
    return "".join(unicode_chars), decoded_labels

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sequence OCR from line image")
    parser.add_argument("--script", type=str, required=True, choices=["jawa", "sunda"])
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    args = parser.parse_args()
    
    save_dir = PROJECT_ROOT / "model" / "saved" / args.script
    model_path = save_dir / "aksara_crnn.tflite"
    labels_path = save_dir / "labels.json"
    
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}. Train the model first.")
        
    # Load labels
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    labels = labels_data["class_names"]
    label_to_char = labels_data["label_to_char"]
    
    # Preprocess image
    input_data = preprocess_image(args.image)
    
    # Load and run TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Decode
    unicode_str, labels_list = decode_prediction(predictions, labels, label_to_char)
    
    print("=" * 60)
    print("  OCR CRNN + CTC Sentence Prediction")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Unicode Output: {unicode_str}")
    print(f"Transliteration: {' '.join(labels_list)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
