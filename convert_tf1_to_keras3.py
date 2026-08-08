#!/usr/bin/env python3
"""
Convert a TensorFlow 1 checkpoint into a Keras3-compatible .keras model file.

This re-uses the TF1->TF2 layer mapping from the repository and instantiates
the MADNet model from `src.madnet`, loads weights from a TF1 checkpoint and
saves the resulting Keras model to `output_path` (recommended to end with
`.keras`).

Example:
  python3 convert_tf1_to_keras3.py \
    --height 480 --width 640 \
    --checkpoint_path /path/to/tf1.ckpt \
    --output_path notebooks/model_weights/flying_things/epoch-from-tf1.keras

Note: this script must be run in the repo root and with an environment that
has TensorFlow and your project's dependencies installed (the same .venv you
use to run training).
"""
import argparse
import tensorflow as tf
import os

from src.madnet import MADNet


layers_map = {
    # mapping from TF1 checkpoint names -> (tf1_weight, tf1_bias)
    "conv1": ("model/gc-read-pyramid/conv1/weights", "model/gc-read-pyramid/conv1/biases"),
    "conv2": ("model/gc-read-pyramid/conv2/weights", "model/gc-read-pyramid/conv2/biases"),
    "conv3": ("model/gc-read-pyramid/conv3/weights", "model/gc-read-pyramid/conv3/biases"),
    "conv4": ("model/gc-read-pyramid/conv4/weights", "model/gc-read-pyramid/conv4/biases"),
    "conv5": ("model/gc-read-pyramid/conv5/weights", "model/gc-read-pyramid/conv5/biases"),
    "conv6": ("model/gc-read-pyramid/conv6/weights", "model/gc-read-pyramid/conv6/biases"),
    "conv7": ("model/gc-read-pyramid/conv7/weights", "model/gc-read-pyramid/conv7/biases"),
    "conv8": ("model/gc-read-pyramid/conv8/weights", "model/gc-read-pyramid/conv8/biases"),
    "conv9": ("model/gc-read-pyramid/conv9/weights", "model/gc-read-pyramid/conv9/biases"),
    "conv10": ("model/gc-read-pyramid/conv10/weights", "model/gc-read-pyramid/conv10/biases"),
    "conv11": ("model/gc-read-pyramid/conv11/weights", "model/gc-read-pyramid/conv11/biases"),
    "conv12": ("model/gc-read-pyramid/conv12/weights", "model/gc-read-pyramid/conv12/biases"),
    # The rest of the mapping for module blocks follows the same pattern as in
    # convert_tf1_to_tf2.py. Add additional mappings here if you need to map
    # more TF1 variables into specific tf.keras layer names.
}


def print_checkpoint_info(save_path):
    reader = tf.train.load_checkpoint(save_path)
    shapes = reader.get_variable_to_shape_map()
    dtypes = reader.get_variable_to_dtype_map()
    print(f"Checkpoint at '{save_path}': variables={len(shapes)}")
    # print a small sample
    for i, key in enumerate(sorted(shapes.keys())[:20], 1):
        print(f"  {i}. {key} shape={shapes[key]} dtype={dtypes[key].name}")


def load_tf1_weights_into_madnet(input_shape, tf1_ckpt):
    """Instantiate a MADNet model and load TF1 checkpoint variables.

    This attempts to set weights by calling `model.get_layer(name).set_weights([...])`.
    Some TF1->TF2 name remapping may be required for custom modules; extend
    `layers_map` above for any additional layers.
    """
    reader = tf.train.load_checkpoint(tf1_ckpt)

    model = MADNet(search_range=2, name="mad_net")

    # Build or call the model so layers exist
    try:
        # MADNet build expects a dict of inputs
        bs = 1
        bshape = {
            "left_input": (bs, input_shape[0], input_shape[1], input_shape[2]),
            "right_input": (bs, input_shape[0], input_shape[1], input_shape[2]),
            "disp_map": (bs, input_shape[0], input_shape[1], 1),
        }
        model.build(bshape)
    except Exception:
        # fallback to a dummy forward pass
        left = tf.zeros((1, input_shape[0], input_shape[1], input_shape[2]), dtype=tf.float32)
        right = tf.zeros_like(left)
        _ = model({"left_input": left, "right_input": right}, training=False)

    print("Model instantiated. Attempting to load mapped TF1 variables...")

    for tf2_layer, (tf1_w_name, tf1_b_name) in layers_map.items():
        try:
            w_tensor = reader.get_tensor(tf1_w_name)
            b_tensor = reader.get_tensor(tf1_b_name)
        except Exception:
            print(f"  TF1 variables for layer '{tf2_layer}' not found in checkpoint: {tf1_w_name} / {tf1_b_name}")
            continue

        try:
            layer = model.get_layer(tf2_layer)
        except Exception:
            print(f"  Could not find layer '{tf2_layer}' in MADNet model; skipping")
            continue

        try:
            print(f"  Setting weights for layer '{tf2_layer}': expected shapes {[arr.shape for arr in layer.get_weights()]} -> TF1 shapes {(w_tensor.shape, b_tensor.shape)}")
            layer.set_weights([w_tensor, b_tensor])
        except Exception as e:
            print(f"  Failed to set weights for layer '{tf2_layer}': {e}")

    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--checkpoint_path", required=True, help="TF1 checkpoint prefix (or path)")
    p.add_argument("--output_path", required=True, help="Path to save the resulting Keras .keras model")
    args = p.parse_args()

    input_shape = (args.height, args.width, 3)

    if not os.path.exists(args.checkpoint_path + ".index") and not os.path.exists(args.checkpoint_path):
        print("Checkpoint not found. Provide a valid TF1 checkpoint prefix or path.")
        return

    print_checkpoint_info(args.checkpoint_path)

    model = load_tf1_weights_into_madnet(input_shape, args.checkpoint_path)

    # Save model in Keras (.keras) format
    out_dir = os.path.dirname(args.output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Saving Keras model to {args.output_path} ...")
    try:
        model.save(args.output_path, include_optimizer=False)
        print("Saved successfully.")
    except Exception as e:
        print("Failed to save model using model.save(); trying save_weights as fallback:")
        print(e)
        try:
            model.save_weights(args.output_path)
            print("Saved weights successfully (weights-only).")
        except Exception as e2:
            print("Final save attempt failed:", e2)


if __name__ == "__main__":
    main()
