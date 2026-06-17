#!/usr/bin/env python3
"""
Generate synthetic sequence (line-level) dataset for Javanese and Sundanese CRNN OCR models.
Renders random text sequences of 3 to 8 characters into 512x64 images.

Enhanced augmentation for robustness:
- Text intensity variation (200-255)
- Background noise (0-30)
- More aggressive blur and gaussian noise
- Elastic distortion
- Random contrast reduction
- Wider font size range (16-40)
- Random scale jitter
- Synthetic weight variation (dilate/erode)
- Random kerning (character-by-character rendering)
- Perspective transform
- Hard negative mining for ha/ya/la confusion
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates, gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.characters import ScriptVocab  # noqa: E402

FONTS_MAP = {
    "jawa": [
        PROJECT_ROOT / "data/fonts/NotoSansJavanese-Regular.ttf",
        PROJECT_ROOT / "data/fonts/NotoSansJavanese-Bold.ttf",
        PROJECT_ROOT / "data/fonts/nyk Ngayogyan New Italic.ttf",
        PROJECT_ROOT / "data/fonts/TuladhaJejegOT-Regular.ttf",
    ],
    "sunda": [
        PROJECT_ROOT / "data/fonts/NotoSansSundanese-Regular.ttf",
        PROJECT_ROOT / "data/fonts/NotoSansSundanese-Bold.ttf",
    ]
}



def elastic_distortion(image: np.ndarray, alpha: float = 8.0, sigma: float = 3.0) -> np.ndarray:
    """Apply elastic distortion to simulate different rendering/writing styles."""
    h, w = image.shape
    # Generate random displacement fields
    dx = gaussian_filter(np.random.randn(h, w) * alpha, sigma)
    dy = gaussian_filter(np.random.randn(h, w) * alpha, sigma)

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    indices = [
        np.clip(y_coords + dy, 0, h - 1).reshape(-1),
        np.clip(x_coords + dx, 0, w - 1).reshape(-1)
    ]

    distorted = map_coordinates(image.astype(np.float64), indices, order=1, mode='constant', cval=0.0)
    return np.clip(distorted.reshape(h, w), 0, 255).astype(np.uint8)


def random_scale_jitter(image: np.ndarray, scale_range: tuple = (0.85, 1.15)) -> np.ndarray:
    """Randomly rescale text within the canvas to simulate size variation."""
    h, w = image.shape
    scale = random.uniform(*scale_range)

    new_h = int(h * scale)
    new_w = int(w * scale)
    if new_h < 8 or new_w < 32:
        return image

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create output canvas of original size
    out = np.zeros((h, w), dtype=np.uint8)

    # Center the resized image onto the output canvas
    y_off = max(0, (h - new_h) // 2)
    x_off = max(0, (w - new_w) // 2)

    # Crop if resized is larger than original
    src_y = max(0, (new_h - h) // 2)
    src_x = max(0, (new_w - w) // 2)

    copy_h = min(new_h - src_y, h - y_off)
    copy_w = min(new_w - src_x, w - x_off)

    out[y_off:y_off + copy_h, x_off:x_off + copy_w] = resized[src_y:src_y + copy_h, src_x:src_x + copy_w]
    return out


def perspective_transform(image: np.ndarray, max_warp: float = 0.02) -> np.ndarray:
    """Apply a light random perspective transform to simulate camera angle variation."""
    h, w = image.shape
    # Define source corners
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Add random perturbation to destination corners
    warp_x = w * max_warp
    warp_y = h * max_warp
    dst = np.float32([
        [random.uniform(0, warp_x), random.uniform(0, warp_y)],
        [w - random.uniform(0, warp_x), random.uniform(0, warp_y)],
        [w - random.uniform(0, warp_x), h - random.uniform(0, warp_y)],
        [random.uniform(0, warp_x), h - random.uniform(0, warp_y)]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


def add_noise(image: np.ndarray) -> np.ndarray:
    """Add random noise augmentation to the image (enhanced version)."""
    # --- Synthetic weight variation (dilate/erode, 30% probability) ---
    if random.random() < 0.30:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        if random.random() < 0.5:
            image = cv2.dilate(image, kernel, iterations=1)
        else:
            image = cv2.erode(image, kernel, iterations=1)

    # --- Random blur (increased probability and kernel variety) ---
    if random.random() < 0.45:
        k = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 2.0)
        image = cv2.GaussianBlur(image, (k, k), sigma)

    # --- Random Gaussian noise (increased probability and range) ---
    if random.random() < 0.45:
        h, w = image.shape
        sigma = random.uniform(5, 25)
        gauss = np.random.normal(0, sigma, (h, w)).astype(np.float32)
        noisy = image.astype(np.float32) + gauss
        image = np.clip(noisy, 0, 255).astype(np.uint8)

    # --- Random thinning/thickening (erosion/dilation) ---
    if random.random() < 0.35:
        kernel_size = random.choice([(2, 2), (3, 3)])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        if random.random() < 0.5:
            image = cv2.dilate(image, kernel, iterations=1)
        else:
            image = cv2.erode(image, kernel, iterations=1)

    # --- Elastic distortion ---
    if random.random() < 0.3:
        alpha = random.uniform(4.0, 12.0)
        sigma = random.uniform(2.0, 4.0)
        image = elastic_distortion(image, alpha=alpha, sigma=sigma)

    # --- Random brightness/contrast shift ---
    if random.random() < 0.4:
        # Contrast factor (0.7 = lower contrast, 1.3 = higher contrast)
        contrast = random.uniform(0.6, 1.4)
        # Brightness shift
        brightness = random.uniform(-15, 15)
        img_f = image.astype(np.float32)
        img_f = img_f * contrast + brightness
        image = np.clip(img_f, 0, 255).astype(np.uint8)

    # --- Random contrast reduction (teach model to handle low-contrast) ---
    if random.random() < 0.2:
        # Reduce the dynamic range: map [0,255] -> [low, high]
        low = random.randint(0, 30)
        high = random.randint(180, 255)
        img_f = image.astype(np.float32) / 255.0
        img_f = img_f * (high - low) + low
        image = np.clip(img_f, 0, 255).astype(np.uint8)

    # --- Random scale jitter ---
    if random.random() < 0.25:
        image = random_scale_jitter(image, scale_range=(0.88, 1.12))

    # --- Perspective transform (15% probability) ---
    if random.random() < 0.15:
        image = perspective_transform(image, max_warp=0.02)

    return image


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by a small angle, filling borders with black."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated


def render_text_with_kerning(
    draw: ImageDraw.Draw,
    chars_unicode: list[str],
    font: ImageFont.FreeTypeFont,
    start_x: int,
    start_y: int,
    text_intensity: int,
    word_boundaries: list[int] | None = None,
) -> None:
    """Render characters one by one with random kerning variation.

    Parameters
    ----------
    draw : ImageDraw.Draw
        PIL drawing context.
    chars_unicode : list[str]
        List of unicode characters to render.
    font : ImageFont.FreeTypeFont
        Font to use for rendering.
    start_x, start_y : int
        Starting coordinates.
    text_intensity : int
        Grayscale intensity for text (0-255).
    word_boundaries : list[int] | None
        Indices in chars_unicode where a word boundary (space) should be inserted.
    """
    cursor_x = start_x
    for idx, ch in enumerate(chars_unicode):
        # Add word spacing at boundaries
        if word_boundaries and idx in word_boundaries:
            cursor_x += random.randint(6, 14)  # word space

        draw.text((cursor_x, start_y), ch, fill=text_intensity, font=font)
        bbox = draw.textbbox((0, 0), ch, font=font)
        char_w = bbox[2] - bbox[0]
        # Random kerning: normal advance ± 0-3px
        kerning_offset = random.randint(-3, 3)
        cursor_x += char_w + kerning_offset


def generate_samples(
    script: str,
    split: str,
    num_samples: int,
    output_dir: Path,
    vocab: ScriptVocab
) -> list[dict]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    font_paths = FONTS_MAP[script]
    # Filter out fonts that don't exist
    font_paths = [p for p in font_paths if p.exists()]
    if not font_paths:
        raise RuntimeError(f"No valid fonts found for script '{script}'")

    metadata = []

    print(f"Generating {num_samples} samples for {script} ({split}) using {len(font_paths)} fonts ...")

    for i in range(num_samples):
        # 1. Generate realistic syllable sequences (max length 24)
        while True:
            chosen_labels = []
            word_texts = []
            word_label_counts = []  # track number of labels per word for kerning

            if script == "jawa":
                consonants = ["ha", "na", "ca", "ra", "ka", "da", "ta", "sa", "wa", "la", "pa", "dha", "ja", "ya", "nya", "ma", "ga", "ba", "tha", "nga"]
                hard_negative_consonants = ["ha", "ya", "la"]
                digits = ["angka_0", "angka_1", "angka_2", "angka_3", "angka_4", "angka_5", "angka_6", "angka_7", "angka_8", "angka_9"]
                sandhangan = ["wulu", "suku", "taling", "pepet", "cecak", "layar", "wignyan", "pangkon"]

                # Hard negative mining: 30% of samples start with ha/ya/la
                use_hard_negative = random.random() < 0.30

                num_words = random.randint(1, 3)
                for word_idx in range(num_words):
                    word_labels = []
                    if random.random() < 0.1:  # 10% chance to generate digits
                        num_digits = random.randint(1, 3)
                        for _ in range(num_digits):
                            word_labels.append(random.choice(digits))
                    else:
                        num_syllables = random.randint(1, 3)
                        for syl_idx in range(num_syllables):
                            # Hard negative: first consonant of first word
                            if use_hard_negative and word_idx == 0 and syl_idx == 0:
                                consonant = random.choice(hard_negative_consonants)
                            else:
                                consonant = random.choice(consonants)
                            word_labels.append(consonant)
                            if random.random() < 0.6:  # 60% chance to add sandhangan
                                s = random.choice(sandhangan)
                                word_labels.append(s)
                                if s == "pangkon":
                                    break
                    chosen_labels.extend(word_labels)
                    word_label_counts.append(len(word_labels))
                    word_texts.append("".join([vocab.label_to_char[lbl] for lbl in word_labels]))

            else:  # sunda
                consonants = [
                    "nga_ka", "nga_qa", "nga_ga", "nga_nga", "nga_ca", "nga_ja", "nga_za", "nga_nya",
                    "nga_ta", "nga_da", "nga_na", "nga_pa", "nga_fa", "nga_va", "nga_ba", "nga_ma",
                    "nga_ya", "nga_ra", "nga_la", "nga_wa", "nga_sa", "nga_xa", "nga_ha"
                ]
                swara = ["swara_a", "swara_i", "swara_u", "swara_e_accent", "swara_o", "swara_e", "swara_eu"]
                digits = ["digit_0", "digit_1", "digit_2", "digit_3", "digit_4", "digit_5", "digit_6", "digit_7", "digit_8", "digit_9"]
                rarangken = [
                    "rarangken_panghulu", "rarangken_pamepet", "rarangken_panolong", "rarangken_panyuku",
                    "rarangken_paneleng", "rarangken_paneuleung", "rarangken_panyecek", "rarangken_panglayar",
                    "rarangken_pangwisad", "rarangken_pamingkal", "rarangken_panyakra", "rarangken_panyiku",
                    "rarangken_paten"
                ]

                num_words = random.randint(1, 3)
                for _ in range(num_words):
                    word_labels = []
                    if random.random() < 0.1:  # 10% chance to generate digits
                        num_digits = random.randint(1, 3)
                        for _ in range(num_digits):
                            word_labels.append(random.choice(digits))
                    else:
                        num_syllables = random.randint(1, 3)
                        for _ in range(num_syllables):
                            if random.random() < 0.2:  # 20% chance of independent vowel
                                word_labels.append(random.choice(swara))
                            else:
                                consonant = random.choice(consonants)
                                word_labels.append(consonant)
                                if random.random() < 0.6:  # 60% chance to add rarangken
                                    r = random.choice(rarangken)
                                    word_labels.append(r)
                                    if r == "rarangken_paten":
                                        break
                    chosen_labels.extend(word_labels)
                    word_label_counts.append(len(word_labels))
                    word_texts.append("".join([vocab.label_to_char[lbl] for lbl in word_labels]))

            # Restrict total label length to 16
            if len(chosen_labels) <= 16:
                break

        # Join words with space for rendering
        text_to_render = " ".join(word_texts)
        unicode_chars = [vocab.label_to_char[lbl] for lbl in chosen_labels]

        # 2. Draw text using PIL
        canvas_w = 256
        canvas_h = 64

        # --- Background noise: random low-intensity background (0-30 range) ---
        bg_intensity = random.randint(0, 30) if split == "train" else 0
        pil_img = Image.new("L", (canvas_w, canvas_h), color=bg_intensity)
        draw = ImageDraw.Draw(pil_img)

        # Select random font and size (expanded range: 16-40)
        font_path = random.choice(font_paths)
        font_size = random.randint(16, 40)
        font = ImageFont.truetype(str(font_path), font_size)

        # --- Text intensity variation: random white intensity (200-255) ---
        text_intensity = random.randint(200, 255) if split == "train" else 255

        # Disable character-by-character split rendering to keep proper Raqm text shaping
        use_kerning = False

        if use_kerning:
            # Render character by character with random kerning
            # Build list of all unicode chars and identify word boundaries
            all_chars_unicode = []
            word_boundaries = set()
            char_cursor = 0
            for w_idx, wt in enumerate(word_texts):
                chars_in_word = list(wt)
                if w_idx > 0:
                    word_boundaries.add(char_cursor)
                all_chars_unicode.extend(chars_in_word)
                char_cursor += len(chars_in_word)

            # Estimate total width for centering
            total_w = 0
            for ch in all_chars_unicode:
                bbox = draw.textbbox((0, 0), ch, font=font)
                total_w += bbox[2] - bbox[0]
            # Add approximate word spaces
            total_w += len(word_texts) * 10

            # If text too wide, reduce font
            if total_w > canvas_w * 0.9:
                font_size = int(font_size * (canvas_w * 0.9) / total_w)
                font = ImageFont.truetype(str(font_path), max(14, font_size))
                # Recalculate width
                total_w = 0
                for ch in all_chars_unicode:
                    bbox = draw.textbbox((0, 0), ch, font=font)
                    total_w += bbox[2] - bbox[0]
                total_w += len(word_texts) * 10

            shift_x = random.randint(-15, 15) if total_w < canvas_w * 0.8 else 0
            shift_y = random.randint(-6, 6)
            start_x = max(5, (canvas_w - total_w) // 2 + shift_x)

            # Get vertical position from a sample character
            sample_bbox = draw.textbbox((0, 0), all_chars_unicode[0] if all_chars_unicode else "A", font=font)
            text_h = sample_bbox[3] - sample_bbox[1]
            start_y = (canvas_h - text_h) // 2 - sample_bbox[1] + shift_y

            render_text_with_kerning(
                draw, all_chars_unicode, font, start_x, start_y,
                text_intensity, word_boundaries=word_boundaries
            )
        else:
            # Standard rendering (original approach)
            # Calculate text bounding box to center it
            bbox = draw.textbbox((0, 0), text_to_render, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # If text is wider than canvas, reduce font size
            if text_w > canvas_w * 0.9:
                font_size = int(font_size * (canvas_w * 0.9) / text_w)
                font = ImageFont.truetype(str(font_path), max(14, font_size))
                bbox = draw.textbbox((0, 0), text_to_render, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

            # Center coordinates with some random shift
            shift_x = random.randint(-15, 15) if text_w < canvas_w * 0.8 else 0
            shift_y = random.randint(-6, 6)

            x = max(10, (canvas_w - text_w) // 2 + shift_x)
            y = (canvas_h - text_h) // 2 - bbox[1] + shift_y

            draw.text((x, y), text_to_render, fill=text_intensity, font=font)

        # Convert to numpy array for CV2 processing
        img_np = np.array(pil_img)

        # 3. Add rotations and noise
        if split == "train":
            if random.random() < 0.6:
                angle = random.uniform(-3.5, 3.5)
                img_np = rotate_image(img_np, angle)
            img_np = add_noise(img_np)

        # Save image file
        filename = f"{split}_{i:05d}.png"
        filepath = split_dir / filename
        cv2.imwrite(str(filepath), img_np)

        # Add metadata entry
        metadata.append({
            "filename": f"{split}/{filename}",
            "labels": chosen_labels,
            "text": "".join(unicode_chars)
        })

    return metadata

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sequence datasets for OCR")
    parser.add_argument("--script", type=str, required=True, choices=["jawa", "sunda"])
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--val-samples", type=int, default=250)
    args = parser.parse_args()




    vocab = ScriptVocab(args.script)
    output_dir = PROJECT_ROOT / "data" / "dataset" / args.script

    # Generate splits
    train_meta = generate_samples(args.script, "train", args.train_samples, output_dir, vocab)
    val_meta = generate_samples(args.script, "val", args.val_samples, output_dir, vocab)

    # Save combined metadata
    combined_meta = {
        "train": train_meta,
        "val": val_meta
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(combined_meta, f, indent=2, ensure_ascii=False)

    print(f"Dataset generated successfully at {output_dir}")

if __name__ == "__main__":
    main()
