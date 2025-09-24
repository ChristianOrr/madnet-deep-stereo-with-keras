import tensorflow as tf
import argparse
from src.madnet import MADNet, colorize_img
from src.preprocessing import StereoDatasetCreator
from src.losses_and_metrics import SSIMLoss
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Script for inferencing MADNet')
parser.add_argument("--left_dir", help='Path to left images folder', required=True)
parser.add_argument("--right_dir", help='Path to right images folder', required=True)
parser.add_argument("--num_adapt", help='Number of modules to adapt, needs to be an integer from 0-6',
                    default=0, type=int, required=False)
parser.add_argument("--mad_mode", help='Module selection method for MAD adaptation. \n'
                                       'Options are "random" and "sequential".',
                    default="random", required=False)
parser.add_argument("--search_range", help='Maximum dispacement (ie. smallest disparity)',
                    default=2, type=int, required=False)
parser.add_argument("-o", "--output_path",
                    help='Path for saving the adapted model (only needed if performing MAD)',
                    default=None, required=False)
parser.add_argument("--weights_path",
                    help='One of the following pretrained weights (will download automatically): '
                         '"synthetic", "kitti", "tf1_conversion_synthetic", "tf1_conversion_kitti"'
                         'or a path to pretrained MADNet weights file (for fine turning)',
                    default=None, required=False)
parser.add_argument("--lr", help="learning rate (only needed if performing adaptation while inferencing.",
                    default=0.0001, type=float, required=False)
parser.add_argument("--height", help='Model image input height resolution', type=int, default=480)
parser.add_argument("--width", help='Model image input height resolution', type=int, default=640)
parser.add_argument("--batch_size", help='Batch size to use during training', type=int, default=1)
parser.add_argument("--show_pred", help='Displays the models predictions', action="store_true", default=False)
parser.add_argument("--steps", help='Number of steps to inference, set to None to inference all the data',
                    default=None, type=int, required=False)

args = parser.parse_args()

def main(args):
    if args.output_path is None and args.num_adapt != 0:
        raise ValueError("No output_path provided for adaptation."
                         "Either set num_adapt=0 or provide a path to output_path"
                         f"Provided args: output_path: {args.output_path}, num_adapt: {args.num_adapt}")

    # Initialise the model
    model = MADNet(
        input_shape=(args.height, args.width, 3),
        # weights=args.weights_path,
        # num_adapt_modules=args.num_adapt,
        # mad_mode=args.mad_mode,
        search_range=args.search_range
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss=SSIMLoss(),
        metrics=None,
        run_eagerly=False
    )
    # Get inferencing data
    predict_dataset = StereoDatasetCreator(
        left_dir=args.left_dir,
        right_dir=args.right_dir,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        shuffle=False,
        disp_dir=None
    )
    predict_ds = predict_dataset()
    # inference the dataset
    disparities = model.predict(predict_ds, steps=args.steps)

    # View disparity predictions
    if args.show_pred:
        for i in range(disparities.shape[0]):
            plt.axis("off")
            plt.grid(visible=None)
            disp = tf.expand_dims(disparities[i, :, :, :], axis=0)
            plt.imshow(colorize_img(disp, cmap='jet')[0])
            plt.show()

    # save the checkpoint and saved_model if it was updated
    if args.num_adapt != 0:
        model.save_weights(args.output_path)


if __name__ == "__main__":
    main(args)