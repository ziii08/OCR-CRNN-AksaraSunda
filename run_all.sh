#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$PROJECT_DIR/venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/venv/bin/python"
elif [ -f "$PROJECT_DIR/../ocr-venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/../ocr-venv/bin/python"
elif [ -f "$PROJECT_DIR/ocr-venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/ocr-venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="python"
else
    PYTHON="python3"
fi

FLUTTER_ASSETS_DIR="$PROJECT_DIR/../OCR-AksaraTest/assets/models"

echo "=========================================================================="
echo " Starting Sundanese Sequence OCR CRNN + CTC Training Pipeline "
echo "  Project Dir: $PROJECT_DIR"
echo "  Python Exec: $PYTHON"
echo "=========================================================================="

# 1. Clean old dataset
echo "Cleaning old Sundanese dataset..."
rm -rf "$PROJECT_DIR/data/dataset/sunda"

# 2. Generate Sundanese dataset (15,000 training, 2,000 validation samples)
echo -e "\n[Step 1/3] Generating Sundanese sentence images..."
$PYTHON "$PROJECT_DIR/data/generate_sequence.py" --script sunda --train-samples 15000 --val-samples 2000

# 3. Train Sundanese model
echo -e "\n[Step 2/3] Training Sundanese CRNN + CTC model..."
$PYTHON "$PROJECT_DIR/model/train.py" --script sunda --epochs 30 --lr 1e-3

# 4. Deploy Sundanese model to Flutter assets
if [ -d "$PROJECT_DIR/../OCR-AksaraTest" ]; then
    echo -e "\n[Step 3/3] Deploying Sundanese model to Flutter assets..."
    mkdir -p "$FLUTTER_ASSETS_DIR"
    cp "$PROJECT_DIR/model/saved/sunda/aksara_crnn.tflite" "$FLUTTER_ASSETS_DIR/aksara_sunda_ocr.tflite"
    cp "$PROJECT_DIR/model/saved/sunda/labels.json" "$FLUTTER_ASSETS_DIR/labels_sunda.json"
    echo "Copied models to Flutter assets."
else
    echo -e "\n[Step 3/3] Flutter project directory not found. Skipping local deployment."
    echo "TFLite model and labels are available at:"
    echo "  $PROJECT_DIR/model/saved/sunda/aksara_crnn.tflite"
    echo "  $PROJECT_DIR/model/saved/sunda/labels.json"
fi

echo -e "\n=========================================================================="
echo " Sundanese training completed successfully!"
echo "=========================================================================="
