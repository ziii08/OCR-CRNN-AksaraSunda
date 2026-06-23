#!/usr/bin/env python3
"""
Training script for Aksara Jawa and Aksara Sunda CRNN + CTC sequence OCR model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.characters import ScriptVocab  # noqa: E402
from model.architecture import create_model  # noqa: E402

class OcrValidationCallback(keras.callbacks.Callback):
    def __init__(self, inference_model, val_dataset, vocab):
        super().__init__()
        self.inference_model = inference_model
        self.val_dataset = val_dataset
        self.vocab = vocab

    def on_epoch_end(self, epoch, logs=None):
        for batch in self.val_dataset.take(1):
            inputs, _ = batch
            images = inputs["image"]
            labels = inputs["label"]
            lengths = inputs["label_length"]
            
            print(f"\n--- Validation predictions after epoch {epoch + 1} ---")
            for i in range(min(5, len(images))):
                img = tf.expand_dims(images[i], axis=0) # [1, 64, 512, 1]
                preds = self.inference_model.predict(img, verbose=0)[0]
                timesteps = preds.shape[0]
                decoded = []
                prev = -1
                blank = self.vocab.num_classes
                for t in range(timesteps):
                    max_idx = int(np.argmax(preds[t]))
                    if max_idx != blank:
                        if max_idx != prev:
                            decoded.append(max_idx)
                        prev = max_idx
                    else:
                        prev = -1
                
                lbl_len = int(lengths[i])
                expected_indices = [int(x) for x in labels[i][:lbl_len]]
                expected_text = self.vocab.decode_to_unicode(expected_indices)
                predicted_text = self.vocab.decode_to_unicode(decoded)
                
                print(f"Sample {i+1} | Expected: {expected_text} | Predicted: {predicted_text} (indices: {decoded})")


class ArtifactExportCallback(keras.callbacks.Callback):
    def __init__(self, save_dir: Path, export_dir: Path | None):
        super().__init__()
        self.save_dir = save_dir
        self.export_dir = export_dir

    def _copy_artifacts(self) -> None:
        if self.export_dir is None:
            return
        self.export_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "latest_checkpoint.weights.h5",
            "crnn_checkpoint.weights.h5",
            "training_log.csv",
            "labels.json",
            "aksara_crnn.keras",
            "aksara_crnn.tflite",
        ]:
            src = self.save_dir / name
            if src.exists():
                shutil.copy2(src, self.export_dir / name)

    def on_epoch_end(self, epoch, logs=None):
        self._copy_artifacts()

    def on_train_end(self, logs=None):
        self._copy_artifacts()


# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 45
IMG_WIDTH = 512
IMG_HEIGHT = 64
MAX_LABEL_LEN = 96  # Maximum labels in one line image; CTC has 128 timesteps at 512px width.

def load_dataset_metadata(script_dir: Path) -> tuple[list[dict], list[dict]]:
    metadata_path = script_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    return meta["train"], meta["val"]

def make_dataset(
    samples: list[dict],
    script_dir: Path,
    vocab: ScriptVocab,
    shuffle: bool = True,
    clear_cache: bool = False,
) -> tf.data.Dataset:
    # Extract paths and encode labels
    img_paths = [str(script_dir / s["filename"]) for s in samples]
    
    encoded_labels = []
    label_lengths = []
    for s in samples:
        indices = vocab.encode_string(s["labels"])
        if len(indices) > MAX_LABEL_LEN:
            raise ValueError(
                f"Label length {len(indices)} exceeds MAX_LABEL_LEN={MAX_LABEL_LEN}: {s.get('filename')}"
            )
        label_lengths.append(len(indices))
        # Pad label sequence to MAX_LABEL_LEN with 0 (ignored by CTC since length is specified)
        padded = indices + [0] * (MAX_LABEL_LEN - len(indices))
        encoded_labels.append(padded)
        
    # Create TF datasets
    path_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(encoded_labels)
    len_ds = tf.data.Dataset.from_tensor_slices(label_lengths)
    
    # Load and preprocess image function
    def process_image(img_path):
        img_bytes = tf.io.read_file(img_path)
        image = tf.io.decode_png(img_bytes, channels=1)
        # Generator already emits fixed-size line images; this also normalizes older datasets.
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        # Force division by 255.0 to normalize to [0.0, 1.0]
        image = tf.cast(image, tf.float32) / 255.0
        return image
        
    img_ds = path_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Zip inputs: CRNN model takes {"image": img, "label": lbl, "label_length": lbl_len}
    dataset = tf.data.Dataset.zip((img_ds, label_ds, len_ds))
    
    def map_to_model_inputs(image, label, length):
        return {
            "image": image,
            "label": tf.cast(label, tf.int32),
            "label_length": tf.cast(length, tf.int32)
        }, tf.zeros_like(length)
        
    dataset = dataset.map(map_to_model_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache preprocessed dataset to disk so epochs 2+ are super fast
    cache_file = script_dir / f"tf_cache_{IMG_WIDTH}x{IMG_HEIGHT}_{'train' if shuffle else 'val'}"
    if clear_cache:
        for p in cache_file.parent.glob(f"{cache_file.name}*"):
            try:
                p.unlink()
            except OSError:
                pass
    dataset = dataset.cache(str(cache_file))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)
        
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

def main() -> None:
    parser = argparse.ArgumentParser(description="Train sequence CRNN model")
    parser.add_argument("--script", type=str, required=True, choices=["jawa", "sunda"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--resume", action="store_true", help="Resume from saved checkpoint weights if available")
    parser.add_argument("--initial-weights", type=str, default="", help="Checkpoint weights path copied from a previous Kaggle run")
    parser.add_argument("--export-dir", type=str, default="", help="Directory to copy checkpoints/logs after each epoch")
    parser.add_argument("--clear-cache", action="store_true", help="Rebuild tf.data cache files before training")
    args = parser.parse_args()
    
    script_dir = PROJECT_ROOT / "data" / "dataset" / args.script
    save_dir = PROJECT_ROOT / "model" / "saved" / args.script
    backup_dir = save_dir / "training_backup"
    export_dir = Path(args.export_dir) if args.export_dir else None
    save_dir.mkdir(parents=True, exist_ok=True)
    
    vocab = ScriptVocab(args.script)
    
    # Load metadata
    train_samples, val_samples = load_dataset_metadata(script_dir)
    
    # Build data loaders
    train_ds = make_dataset(train_samples, script_dir, vocab, shuffle=True, clear_cache=args.clear_cache)
    val_ds = make_dataset(val_samples, script_dir, vocab, shuffle=False, clear_cache=args.clear_cache)
    
    # Build model
    print("\nBuilding CRNN Model ...")
    training_model, inference_model, base_model = create_model(
        num_classes=vocab.num_classes,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT
    )
    
    # Compile model (loss is computed internally in train_step)
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr)
    )
    # Subclassed Keras models must be built before BackupAndRestore or load_weights.
    training_model(tf.zeros((1, IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32))

    best_checkpoint_path = save_dir / "crnn_checkpoint.weights.h5"
    latest_checkpoint_path = save_dir / "latest_checkpoint.weights.h5"
    initial_weights_path = Path(args.initial_weights) if args.initial_weights else None
    if initial_weights_path and initial_weights_path.exists():
        print(f"\nLoading initial weights from {initial_weights_path} ...")
        training_model.load_weights(str(initial_weights_path))
    elif args.resume and latest_checkpoint_path.exists():
        print(f"\nResuming latest checkpoint weights from {latest_checkpoint_path} ...")
        training_model.load_weights(str(latest_checkpoint_path))
    elif args.resume and best_checkpoint_path.exists():
        print(f"\nResuming best checkpoint weights from {best_checkpoint_path} ...")
        training_model.load_weights(str(best_checkpoint_path))
    elif best_checkpoint_path.exists():
        print(f"\nCheckpoint exists but --resume was not passed: {best_checkpoint_path}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(latest_checkpoint_path),
            save_best_only=False,
            save_weights_only=True,
            verbose=1
        ),
        keras.callbacks.BackupAndRestore(
            backup_dir=str(backup_dir)
        ),
        keras.callbacks.CSVLogger(
            filename=str(save_dir / "training_log.csv"),
            append=args.resume
        ),
        OcrValidationCallback(inference_model, val_ds, vocab),
        ArtifactExportCallback(save_dir, export_dir)
    ]
    
    # Train
    print("\nStarting Training ...")
    training_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Load best weights explicitly if they exist before saving
    if best_checkpoint_path.exists():
        print(f"Loading best checkpoint weights from {best_checkpoint_path}")
        training_model.load_weights(str(best_checkpoint_path))
        
    # Save training weights
    keras_path = save_dir / "aksara_crnn.keras"
    base_model.save(str(keras_path))
    print(f"Base Keras model saved to {keras_path}")
    
    # Export Inference model to TFLite
    print("\nConverting inference model to TFLite ...")
    tflite_path = save_dir / "aksara_crnn.tflite"
    
    try:
        # Try converting directly first (works if CPU-only or no GPU-specific ops initialized)
        converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        tflite_model = converter.convert()
        tflite_path.write_bytes(tflite_model)
        print(f"TFLite model saved to {tflite_path}  ({len(tflite_model) / 1024:.1f} KB)")
    except Exception as e:
        print(f"Direct conversion failed ({e}). Spawning CPU-only subprocess for TFLite conversion...")
        import os
        import subprocess
        
        conversion_code = f"""
import sys
import tensorflow as tf
from pathlib import Path
sys.path.insert(0, '{PROJECT_ROOT}')
from data.characters import ScriptVocab
from model.architecture import create_model

vocab = ScriptVocab('{args.script}')
training_model, inference_model, _ = create_model(
    num_classes=vocab.num_classes,
    img_width={IMG_WIDTH},
    img_height={IMG_HEIGHT}
)
training_model(tf.zeros((1, {IMG_HEIGHT}, {IMG_WIDTH}, 1)))
training_model.load_weights('{best_checkpoint_path}')

converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
Path('{tflite_path}').write_bytes(tflite_model)
print("TFLite conversion in CPU subprocess succeeded!")
"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run([sys.executable, "-c", conversion_code], env=env, check=True)
        print(f"TFLite model saved successfully at {tflite_path}")
    
    # Save vocab mapping JSON for Flutter app
    labels_mapping = {
        "class_names": vocab.labels,
        "label_to_char": vocab.label_to_char,
        "index_to_label": vocab.idx_to_label
    }
    with open(save_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_mapping, f, indent=2, ensure_ascii=False)
    print("Label maps exported successfully.")

if __name__ == "__main__":
    main()
