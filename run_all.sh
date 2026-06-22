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
SCRIPT="sunda"
DATASET_DIR="$PROJECT_DIR/data/dataset/$SCRIPT"
SAVE_DIR="$PROJECT_DIR/model/saved/$SCRIPT"
KAGGLE_OUTPUT_DIR="/kaggle/working/aksara_sunda_ocr_outputs"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-80000}"
VAL_SAMPLES="${VAL_SAMPLES:-5000}"
EPOCHS="${EPOCHS:-45}"
LR="${LR:-5e-4}"
RESUME="${RESUME:-1}"
CLEAN_DATASET="${CLEAN_DATASET:-0}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
CLEAR_CACHE="${CLEAR_CACHE:-0}"

echo "=========================================================================="
echo " Starting Sundanese Sequence OCR CRNN + CTC Training Pipeline "
echo "  Project Dir: $PROJECT_DIR"
echo "  Python Exec: $PYTHON"
echo "  Train Samples: $TRAIN_SAMPLES"
echo "  Val Samples: $VAL_SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Resume: $RESUME"
echo "=========================================================================="

# 1. Optionally clean old dataset
if [ "$CLEAN_DATASET" = "1" ]; then
    echo "Cleaning old Sundanese dataset..."
    rm -rf "$DATASET_DIR"
fi

# 2. Generate Sundanese dataset (80,000 training, 5,000 validation samples)
if [ "$FORCE_REGENERATE" = "1" ] || [ ! -f "$DATASET_DIR/metadata.json" ]; then
    echo -e "\n[Step 1/3] Generating Sundanese sentence images..."
    $PYTHON "$PROJECT_DIR/data/generate_sequence.py" --script "$SCRIPT" --train-samples "$TRAIN_SAMPLES" --val-samples "$VAL_SAMPLES"
else
    echo -e "\n[Step 1/3] Reusing existing Sundanese dataset at $DATASET_DIR"
    echo "Set FORCE_REGENERATE=1 to rebuild it."
fi

# 3. Train Sundanese model
echo -e "\n[Step 2/3] Training Sundanese CRNN + CTC model..."
TRAIN_ARGS=("--script" "$SCRIPT" "--epochs" "$EPOCHS" "--lr" "$LR")
if [ "$RESUME" = "1" ]; then
    TRAIN_ARGS+=("--resume")
fi
if [ "$CLEAR_CACHE" = "1" ]; then
    TRAIN_ARGS+=("--clear-cache")
fi
$PYTHON "$PROJECT_DIR/model/train.py" "${TRAIN_ARGS[@]}"

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

if [ -d "/kaggle/working" ]; then
    echo -e "\nCopying Sundanese training outputs to $KAGGLE_OUTPUT_DIR ..."
    mkdir -p "$KAGGLE_OUTPUT_DIR"
    cp "$SAVE_DIR/aksara_crnn.tflite" "$KAGGLE_OUTPUT_DIR/" 2>/dev/null || true
    cp "$SAVE_DIR/aksara_crnn.keras" "$KAGGLE_OUTPUT_DIR/" 2>/dev/null || true
    cp "$SAVE_DIR/labels.json" "$KAGGLE_OUTPUT_DIR/" 2>/dev/null || true
    cp "$SAVE_DIR/crnn_checkpoint.weights.h5" "$KAGGLE_OUTPUT_DIR/" 2>/dev/null || true
    cp "$SAVE_DIR/training_log.csv" "$KAGGLE_OUTPUT_DIR/" 2>/dev/null || true
fi

echo -e "\n=========================================================================="
echo " Sundanese training completed successfully!"
echo "=========================================================================="
