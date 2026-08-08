#!/usr/bin/env python3
"""Load an HDF5 weights-only file into the MADNet model (Keras3-compatible).

Usage: run from repo root with the project's virtualenv python:
  .venv/bin/python3 notebooks/load_h5_to_madnet.py \
    --weights notebooks/model_weights/flying_things/synthetic.h5

This script will:
 - import MADNet from `src.madnet`
 - instantiate the model
 - run a dummy forward pass to ensure variables exist
 - call `model.load_weights(weights_path, by_name=True)` to load matching weights
 - print a concise verification of loaded weights
"""
import os
import sys
import argparse
import traceback

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import tensorflow as tf
import keras
from src.madnet import MADNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=False,
                   default=os.path.join(repo_root, "notebooks/model_weights/flying_things/synthetic.h5"),
                   help="Path to HDF5 weights-only file")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()

    weights_path = args.weights
    print("weights_path:", weights_path)
    if not os.path.exists(weights_path):
        print("Weights file not found:", weights_path)
        return 2

    try:
        print("Instantiating MADNet model...")
        model = MADNet(name="mad_net")

        # Force variable creation via a dummy forward pass (preferred over model.build()
        # because MADNet may add sublayers during build which can cause tracking errors
        # if the layer is already built).
        h, w = args.height, args.width
        batch = args.batch
        left = tf.zeros((batch, h, w, 3), dtype=tf.float32)
        right = tf.zeros_like(left)
        print("Running dummy forward pass to create variables...")
        _ = model({"left_input": left, "right_input": right}, training=False)

        print("Variables created. Attempting to load weights by_name=True...")
        # by_name=True is useful when HDF5 layer names differ in ordering
        model.load_weights(weights_path, by_name=True)

        print("Weights loaded — verifying some tensors:")
        weights = model.weights
        print(f"Total weight tensors: {len(weights)}")
        # Print first 10 weight names/shapes
        for i, w in enumerate(weights[:10], 1):
            try:
                arr = w.numpy()
                print(f"{i}: {w.name} shape={arr.shape} dtype={arr.dtype} preview={arr.ravel()[:6]}")
            except Exception:
                print(f"{i}: {getattr(w,'name',w)} (couldn't read numpy)")

        print("Done — if you need the full list, inspect `model.weights` in a notebook.")
        return 0

    except Exception:
        print("Exception while loading weights:")
        traceback.print_exc()
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
