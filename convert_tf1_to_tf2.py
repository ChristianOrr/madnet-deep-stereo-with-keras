import argparse
import tensorflow as tf
from src.madnet import MADNet

parser = argparse.ArgumentParser(description='Script for loading the tf1 pretrained '
                                             'checkpoint into the tf2/keras model.')
parser.add_argument("--height", help='Model image input/output height resolution', type=int, default=480)
parser.add_argument("--width", help='Model image input/output height resolution', type=int, default=640)
parser.add_argument("-o", "--output_path",
                    help='Path to save TF2 model converted from the TF1 pretrained checkpoint',
                    required=True)
parser.add_argument("--checkpoint_path", help="path to pretrained TensorFlow 1 checkpoint",  required=True)
args = parser.parse_args()


def print_checkpoint(save_path):
    """
    Displays the checkpoint layers shape, dtype and values
    """
    reader = tf.train.load_checkpoint(save_path)
    shapes = reader.get_variable_to_shape_map()
    dtypes = reader.get_variable_to_dtype_map()
    print(f"Checkpoint at '{save_path}':")
    for key in shapes:
        print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}, "
            f"value={reader.get_tensor(key)})")


def load_tf1_weights(input_shape, tf1_weights):
    """
    Takes the tf1 pretrained model and loads the
    weights/biases into the tf2 keras functional model

    """
    # Mapping of tf2 functional model trainable layer names
    # to their respective weights/biases names in the tf1 checkpoint
    layers_map = {
        "conv1": ("model/gc-read-pyramid/conv1/weights",
                  "model/gc-read-pyramid/conv1/biases"),
        "conv2": ("model/gc-read-pyramid/conv2/weights",
                  "model/gc-read-pyramid/conv2/biases"),
        "conv3": ("model/gc-read-pyramid/conv3/weights",
                  "model/gc-read-pyramid/conv3/biases"),
        "conv4": ("model/gc-read-pyramid/conv4/weights",
                  "model/gc-read-pyramid/conv4/biases"),
        "conv5": ("model/gc-read-pyramid/conv5/weights",
                  "model/gc-read-pyramid/conv5/biases"),
        "conv6": ("model/gc-read-pyramid/conv6/weights",
                  "model/gc-read-pyramid/conv6/biases"),
        "conv7": ("model/gc-read-pyramid/conv7/weights",
                  "model/gc-read-pyramid/conv7/biases"),
        "conv8": ("model/gc-read-pyramid/conv8/weights",
                  "model/gc-read-pyramid/conv8/biases"),
        "conv9": ("model/gc-read-pyramid/conv9/weights",
                  "model/gc-read-pyramid/conv9/biases"),
        "conv10": ("model/gc-read-pyramid/conv10/weights",
                   "model/gc-read-pyramid/conv10/biases"),
        "conv11": ("model/gc-read-pyramid/conv11/weights",
                   "model/gc-read-pyramid/conv11/biases"),
        "conv12": ("model/gc-read-pyramid/conv12/weights",
                   "model/gc-read-pyramid/conv12/biases"),
        "volume_filtering_6_disp1": ("model/G6/fgc-volume-filtering-6/disp-1/weights",
                                     "model/G6/fgc-volume-filtering-6/disp-1/biases"),
        "volume_filtering_6_disp2": ("model/G6/fgc-volume-filtering-6/disp-2/weights",
                                     "model/G6/fgc-volume-filtering-6/disp-2/biases"),
        "volume_filtering_6_disp3": ("model/G6/fgc-volume-filtering-6/disp-3/weights",
                                     "model/G6/fgc-volume-filtering-6/disp-3/biases"),
        "volume_filtering_6_disp4": ("model/G6/fgc-volume-filtering-6/disp-4/weights",
                                     "model/G6/fgc-volume-filtering-6/disp-4/biases"),
        "volume_filtering_6_disp5": ("model/G6/fgc-volume-filtering-6/disp-5/weights",
                                     "model/G6/fgc-volume-filtering-6/disp-5/biases"),
        "volume_filtering_6_disp6": ("model/G6/fgc-volume-filtering-6/disp-6/weights",
                                     "model/G6/fgc-volume-filtering-6/disp-6/biases"),
        "volume_filtering_5_disp1": ("model/G5/fgc-volume-filtering-5/disp-1/weights",
                                     "model/G5/fgc-volume-filtering-5/disp-1/biases"),
        "volume_filtering_5_disp2": ("model/G5/fgc-volume-filtering-5/disp-2/weights",
                                     "model/G5/fgc-volume-filtering-5/disp-2/biases"),
        "volume_filtering_5_disp3": ("model/G5/fgc-volume-filtering-5/disp-3/weights",
                                     "model/G5/fgc-volume-filtering-5/disp-3/biases"),
        "volume_filtering_5_disp4": ("model/G5/fgc-volume-filtering-5/disp-4/weights",
                                     "model/G5/fgc-volume-filtering-5/disp-4/biases"),
        "volume_filtering_5_disp5": ("model/G5/fgc-volume-filtering-5/disp-5/weights",
                                     "model/G5/fgc-volume-filtering-5/disp-5/biases"),
        "volume_filtering_5_disp6": ("model/G5/fgc-volume-filtering-5/disp-6/weights",
                                     "model/G5/fgc-volume-filtering-5/disp-6/biases"),
        "volume_filtering_4_disp1": ("model/G4/fgc-volume-filtering-4/disp-1/weights",
                                     "model/G4/fgc-volume-filtering-4/disp-1/biases"),
        "volume_filtering_4_disp2": ("model/G4/fgc-volume-filtering-4/disp-2/weights",
                                     "model/G4/fgc-volume-filtering-4/disp-2/biases"),
        "volume_filtering_4_disp3": ("model/G4/fgc-volume-filtering-4/disp-3/weights",
                                     "model/G4/fgc-volume-filtering-4/disp-3/biases"),
        "volume_filtering_4_disp4": ("model/G4/fgc-volume-filtering-4/disp-4/weights",
                                     "model/G4/fgc-volume-filtering-4/disp-4/biases"),
        "volume_filtering_4_disp5": ("model/G4/fgc-volume-filtering-4/disp-5/weights",
                                     "model/G4/fgc-volume-filtering-4/disp-5/biases"),
        "volume_filtering_4_disp6": ("model/G4/fgc-volume-filtering-4/disp-6/weights",
                                     "model/G4/fgc-volume-filtering-4/disp-6/biases"),
        "volume_filtering_3_disp1": ("model/G3/fgc-volume-filtering-3/disp-1/weights",
                                     "model/G3/fgc-volume-filtering-3/disp-1/biases"),
        "volume_filtering_3_disp2": ("model/G3/fgc-volume-filtering-3/disp-2/weights",
                                     "model/G3/fgc-volume-filtering-3/disp-2/biases"),
        "volume_filtering_3_disp3": ("model/G3/fgc-volume-filtering-3/disp-3/weights",
                                     "model/G3/fgc-volume-filtering-3/disp-3/biases"),
        "volume_filtering_3_disp4": ("model/G3/fgc-volume-filtering-3/disp-4/weights",
                                     "model/G3/fgc-volume-filtering-3/disp-4/biases"),
        "volume_filtering_3_disp5": ("model/G3/fgc-volume-filtering-3/disp-5/weights",
                                     "model/G3/fgc-volume-filtering-3/disp-5/biases"),
        "volume_filtering_3_disp6": ("model/G3/fgc-volume-filtering-3/disp-6/weights",
                                     "model/G3/fgc-volume-filtering-3/disp-6/biases"),
        "volume_filtering_2_disp1": ("model/G2/fgc-volume-filtering-2/disp-1/weights",
                                     "model/G2/fgc-volume-filtering-2/disp-1/biases"),
        "volume_filtering_2_disp2": ("model/G2/fgc-volume-filtering-2/disp-2/weights",
                                     "model/G2/fgc-volume-filtering-2/disp-2/biases"),
        "volume_filtering_2_disp3": ("model/G2/fgc-volume-filtering-2/disp-3/weights",
                                     "model/G2/fgc-volume-filtering-2/disp-3/biases"),
        "volume_filtering_2_disp4": ("model/G2/fgc-volume-filtering-2/disp-4/weights",
                                     "model/G2/fgc-volume-filtering-2/disp-4/biases"),
        "volume_filtering_2_disp5": ("model/G2/fgc-volume-filtering-2/disp-5/weights",
                                     "model/G2/fgc-volume-filtering-2/disp-5/biases"),
        "volume_filtering_2_disp6": ("model/G2/fgc-volume-filtering-2/disp-6/weights",
                                     "model/G2/fgc-volume-filtering-2/disp-6/biases"),
        "context1": ("model/context-1/weights",
                     "model/context-1/biases"),
        "context2": ("model/context-2/weights",
                     "model/context-2/biases"),
        "context3": ("model/context-3/weights",
                     "model/context-3/biases"),
        "context4": ("model/context-4/weights",
                     "model/context-4/biases"),
        "context5": ("model/context-5/weights",
                     "model/context-5/biases"),
        "context6": ("model/context-6/weights",
                     "model/context-6/biases"),
        "context7": ("model/context-7/weights",
                     "model/context-7/biases")
    }

    # load tf1 checkpoint
    reader = tf.train.load_checkpoint(tf1_weights)

    # load tf2 functional model
    model = MADNet(
        input_shape=input_shape,
    )

    for tf2_layer, (tf1_weight, tf1_bias) in layers_map.items():
        print(f"\n\nLoading weights/biases for layer: {tf2_layer}")
        print(f"\nWeights before: {model.get_layer(tf2_layer).get_weights()}")
        model.get_layer(tf2_layer).set_weights([reader.get_tensor(tf1_weight), reader.get_tensor(tf1_bias)])
        print(f"\nWeights after: {model.get_layer(tf2_layer).get_weights()}")

    return model


def load_and_save(input_shape, tf1_weights, output_path):
    """
    Loads the tf1 checkpoint into the tf2 functional model
    and saves the tf2 model with the pretrained weights
    to the desired location.
    """
    model = load_tf1_weights(input_shape, tf1_weights)
    print("\nSaving model weights")
    model.save_weights(
        filepath=output_path
    )
    print("Saving Complete!")

if __name__ == "__main__":
    input_shape = (args.height, args.width, 3)
    load_and_save(
        input_shape,
        args.checkpoint_path,
        args.output_path
    )

