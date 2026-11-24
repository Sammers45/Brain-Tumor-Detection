"""Stage 4: model definitions.

- cnn       : main model. Small 3-block CNN + dropout-regularised dense head,
              sized for a 253-image dataset.
- lstm      : baseline that reads the image row-by-row as a sequence with a
              bidirectional LSTM. Included to test, with data, whether a
              sequence model is a sensible choice for a single 2D MRI slice.
- transfer  : MobileNetV2 pretrained on ImageNet, fine-tuned. Usually the best
              option for small medical-image datasets; needs internet to fetch
              weights on first run.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def augmentation() -> tf.keras.Sequential:
    # Active only when training=True, so inference is unaffected.
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomContrast(0.1),
        ],
        name="augment",
    )


def build_cnn(input_shape, augment=True) -> tf.keras.Model:
    """Deliberately small CNN (~1.07M params, ~1M of them in the dense head).

    Chosen by a controlled comparison on CV folds (see README "Model design
    decisions"): a deeper CNN with BatchNorm + global average pooling collapsed
    to a constant output on ~135 training images (BatchNorm moving statistics
    never settle in ~9 steps/epoch), and reached only ~0.52 AUC. This lighter
    design learned reliably (~0.79 AUC on the same folds).
    """
    inp = layers.Input(input_shape)
    x = augmentation()(inp) if augment else inp
    for filters in (16, 32, 64):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name="cnn")


def build_lstm(input_shape, augment=True) -> tf.keras.Model:
    h, w, _ = input_shape
    inp = layers.Input(input_shape)
    x = augmentation()(inp) if augment else inp
    x = layers.Reshape((h, w))(x)  # each image row is one timestep
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name="lstm")


def build_transfer(input_shape, augment=True, weights="imagenet") -> tf.keras.Model:
    inp = layers.Input(input_shape)
    x = augmentation()(inp) if augment else inp
    x = layers.Concatenate()([x, x, x])  # grayscale -> 3 channels
    x = layers.Rescaling(2.0, offset=-1.0)(x)  # [0,1] -> [-1,1] as MobileNetV2 expects
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights=weights, input_shape=(input_shape[0], input_shape[1], 3)
    )
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name="transfer_mobilenetv2")


BUILDERS = {"cnn": build_cnn, "lstm": build_lstm, "transfer": build_transfer}


def build(name: str, input_shape, augment=True, lr=1e-3) -> tf.keras.Model:
    model = BUILDERS[name](input_shape, augment=augment)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model
