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

IMG_WIDTH = 256
IMG_HEIGHT = 32

def preprocess_image(image_path: str) -> np.ndarray:
    """Load image, binarize, invert if necessary, and resize to 256x32."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
        
    # Check if background is bright (white paper with dark ink)
    avg = np.mean(img)
    is_inverted = avg > 127
    
    # Binarize
    threshold = 127
    if is_inverted:
        # Dark pixels become white (255), light becomes black (0)
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        # Bright pixels remain white
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        
    # Resize with aspect ratio preservation (padding)
    h, w = binary.shape
    scale = IMG_HEIGHT / h
    new_w = int(w * scale)
    
    if new_w > IMG_WIDTH:
        # Resize to fit width exactly
        resized = cv2.resize(binary, (IMG_WIDTH, IMG_HEIGHT))
    else:
        # Resize height and pad width with black on the right
        resized_h = cv2.resize(binary, (new_w, IMG_HEIGHT))
        padded = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        padded[:, :new_w] = resized_h
        resized = padded
        
    # Normalize to [0, 1] and add batch/channel dims
    input_data = resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=(0, -1)) # [1, 32, 256, 1]
    
    return input_data

def decode_prediction(preds: np.ndarray, labels: list[str], label_to_char: dict) -> tuple[str, list[str]]:
    """Greedy decode predictions from the TFLite output probability matrix."""
    # preds has shape [1, 64, NumClasses + 1]
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
