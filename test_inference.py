import json
import numpy as np
import tensorflow as tf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    model_dir = PROJECT_ROOT / "model" / "saved" / "sunda"
    tflite_path = model_dir / "aksara_crnn.tflite"
    labels_path = model_dir / "labels.json"

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    class_names = labels_data["class_names"]
    label_to_char = labels_data["label_to_char"]

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dataset_dir = PROJECT_ROOT / "data" / "dataset" / "sunda"
    with open(dataset_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    samples = metadata["train"][:5]
    correct = 0
    total = 0

    for s in samples:
        img_path = dataset_dir / s["filename"]
        img_bytes = tf.io.read_file(str(img_path))
        image = tf.io.decode_png(img_bytes, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [32, 256])
        image_np = np.expand_dims(image.numpy(), axis=0)

        interpreter.set_tensor(input_details[0]['index'], image_np)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        pred_probs = output_data[0]
        num_timesteps = pred_probs.shape[0]
        decoded_indices = []
        prev_idx = -1
        blank_idx = len(class_names)

        for t in range(num_timesteps):
            max_idx = int(np.argmax(pred_probs[t]))
            if max_idx != blank_idx:
                if max_idx != prev_idx:
                    decoded_indices.append(max_idx)
                prev_idx = max_idx
            else:
                prev_idx = -1

        decoded_labels = [class_names[idx] for idx in decoded_indices]
        decoded_chars = [label_to_char.get(lbl, lbl) for lbl in decoded_labels]
        expected = s['text']
        predicted = ''.join(decoded_chars)
        match = "✓" if expected == predicted else "✗"
        if expected == predicted:
            correct += 1
        total += 1

        print(f"{match} Expected: {expected}  |  Predicted: {predicted}  |  Labels: {decoded_labels}")
    
    print(f"\nAccuracy: {correct}/{total}")

if __name__ == "__main__":
    main()
