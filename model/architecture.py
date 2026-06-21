"""
CRNN (Convolutional Recurrent Neural Network) model architecture for sequence OCR.
Incorporates a custom OCRModel subclass for training with CTC Loss.

Upgraded architecture:
- Input: 512x64 (from 256x32)
- 6 CNN blocks for higher resolution handling
- 2-layer BiLSTM(256) with dropout
- 128 output timesteps
- Dense projection 256
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class OCRModel(keras.Model):
    """Custom Keras Model that handles CTC Loss calculation and gradient updates."""
    def __init__(self, base_model: keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs):
        # Standard forward pass
        return self.base_model(inputs)

    def train_step(self, data):
        # data is a tuple of (inputs, dummy_targets)
        # inputs is a dict of {"image": img, "label": lbl, "label_length": len}
        inputs, _ = data
        images = inputs["image"]
        labels = inputs["label"]
        label_length = inputs["label_length"]

        with tf.GradientTape() as tape:
            # 1. Forward pass to get logits [batch_size, time_steps, num_classes + 1]
            logits = self(images, training=True)
            
            # 2. Compute CTC loss using TF's optimized implementation
            batch_len = tf.cast(tf.shape(labels)[0], dtype="int32")
            input_length = tf.cast(tf.shape(logits)[1], dtype="int32")
            logit_length = input_length * tf.ones(shape=(batch_len,), dtype="int32")
            
            squeezed_label_length = tf.cast(label_length, dtype="int32")
            if len(squeezed_label_length.shape) == 2:
                squeezed_label_length = tf.squeeze(squeezed_label_length, axis=-1)
            
            loss = tf.nn.ctc_loss(
                labels=tf.cast(labels, dtype="int32"),
                logits=logits,
                label_length=squeezed_label_length,
                logit_length=logit_length,
                logits_time_major=False,
                blank_index=int(logits.shape[2]) - 1
            )
            mean_loss = tf.reduce_mean(loss)

        # 3. Compute gradients and apply
        grads = tape.gradient(mean_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # 4. Update and return loss metric
        self.loss_tracker.update_state(mean_loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        inputs, _ = data
        images = inputs["image"]
        labels = inputs["label"]
        label_length = inputs["label_length"]

        # Forward pass
        logits = self(images, training=False)
        
        # Compute loss
        batch_len = tf.cast(tf.shape(labels)[0], dtype="int32")
        input_length = tf.cast(tf.shape(logits)[1], dtype="int32")
        logit_length = input_length * tf.ones(shape=(batch_len,), dtype="int32")
        
        squeezed_label_length = tf.cast(label_length, dtype="int32")
        if len(squeezed_label_length.shape) == 2:
            squeezed_label_length = tf.squeeze(squeezed_label_length, axis=-1)
        
        loss = tf.nn.ctc_loss(
            labels=tf.cast(labels, dtype="int32"),
            logits=logits,
            label_length=squeezed_label_length,
            logit_length=logit_length,
            logits_time_major=False,
            blank_index=int(logits.shape[2]) - 1
        )
        mean_loss = tf.reduce_mean(loss)

        self.loss_tracker.update_state(mean_loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

def create_model(
    num_classes: int,
    img_width: int = 512,
    img_height: int = 64,
) -> tuple[keras.Model, keras.Model, keras.Model]:
    """Build and return the training, inference, and base CRNN models.

    Parameters
    ----------
    num_classes : int
        Number of characters in vocabulary.
    img_width : int
        Width of input line images (default 512).
    img_height : int
        Height of input line images (default 64).

    Returns
    -------
    tuple[keras.Model, keras.Model, keras.Model]
        (training_model, inference_model, base_model)
    """
    # ── Base CRNN Model (Dynamic Batch) ────────────────────────────────
    base_input = layers.Input(shape=(img_height, img_width, 1), name="base_image", dtype="float32")

    # CNN Block 1: Conv 32 (3,3) + BN + ReLU + MaxPool (2,2)
    # Input: [batch, 64, 512, 1] -> Output: [batch, 32, 256, 32]
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1")(base_input)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # CNN Block 2: Conv 64 (3,3) + BN + ReLU + MaxPool (2,2)
    # Input: [batch, 32, 256, 32] -> Output: [batch, 16, 128, 64]
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # CNN Block 3: Conv 128 (3,3) + BN + ReLU + MaxPool (2,1)
    # Input: [batch, 16, 128, 64] -> Output: [batch, 8, 128, 128]
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2, 1), name="pool3")(x)

    # CNN Block 4: Conv 128 (3,3) + BN + ReLU + MaxPool (2,1)
    # Input: [batch, 8, 128, 128] -> Output: [batch, 4, 128, 128]
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D((2, 1), name="pool4")(x)

    # CNN Block 5: Conv 256 (3,3) + BN + ReLU + MaxPool (2,1)
    # Input: [batch, 4, 128, 128] -> Output: [batch, 2, 128, 256]
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv5")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.ReLU(name="relu5")(x)
    x = layers.MaxPooling2D((2, 1), name="pool5")(x)

    # CNN Block 6: Conv 256 (3,3) + BN + ReLU + MaxPool (2,1)
    # Input: [batch, 2, 128, 256] -> Output: [batch, 1, 128, 256]
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv6")(x)
    x = layers.BatchNormalization(name="bn6")(x)
    x = layers.ReLU(name="relu6")(x)
    x = layers.MaxPooling2D((2, 1), name="pool6")(x)

    if img_height % 64 != 0 or img_width % 4 != 0:
        raise ValueError("CRNN expects img_height divisible by 64 and img_width divisible by 4")

    time_steps = img_width // 4
    feature_dim = (img_height // 64) * 256

    # Reshape CNN features to a left-to-right sequence.
    x = layers.Permute((2, 1, 3), name="permute")(x)
    x = layers.Reshape(target_shape=(time_steps, feature_dim), name="reshape")(x)

    # Dense projection before RNN (256 units)
    x = layers.Dense(256, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense1")(x)
    x = layers.ReLU(name="relu_dense1")(x)
    x = layers.Dropout(0.3, name="dropout1")(x)

    # RNN: 1-layer Bidirectional LSTM(128) with dropout
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="lstm1")(x)

    # Output logits (activation=None)
    y_pred = layers.Dense(num_classes + 1, activation=None, name="dense2")(x)

    base_model = keras.Model(inputs=base_input, outputs=y_pred, name="crnn_base")

    # ── Training Model (Subclassed OCRModel) ──────────────────────────
    training_model = OCRModel(base_model=base_model, name="ocr_training")

    # ── Inference Model (Fixed batch size of 1 for TFLite compilation) ──
    inference_img = layers.Input(batch_shape=(1, img_height, img_width, 1), name="image", dtype="float32")
    inference_logits = base_model(inference_img)
    # Apply softmax only here for inference decoding!
    inference_preds = layers.Softmax(name="softmax")(inference_logits)

    inference_model = keras.Model(inputs=inference_img, outputs=inference_preds, name="crnn_inference")

    return training_model, inference_model, base_model

def ctc_decode(predictions, max_length: int = 96) -> list[list[int]]:
    """Greedy decode predictions using TensorFlow's CTC decoder.

    Parameters
    ----------
    predictions : np.ndarray
        Softmax matrix from model prediction, shape [Batch, TimeSteps, NumClasses+1]
    max_length : int
        Maximum sequence length allowed.

    Returns
    -------
    list[list[int]]
        List of decoded integer label sequences.
    """
    input_len = tf.cast(tf.shape(predictions)[1], dtype="int32")
    batch_len = tf.cast(tf.shape(predictions)[0], dtype="int32")
    
    input_length = input_len * tf.ones(shape=(batch_len,), dtype="int32")
    
    # Use greedy decoder
    decoded, _ = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(predictions, perm=[1, 0, 2]),
        sequence_length=input_length,
        merge_repeated=True
    )
    
    # Extract sparse tensor values
    sparse_decoded = decoded[0]
    dense_decoded = tf.sparse.to_dense(sparse_decoded, default_value=-1).numpy()
    
    results = []
    for seq in dense_decoded:
        # Filter out padding (-1)
        results.append([int(char_idx) for char_idx in seq if char_idx != -1])
    return results
