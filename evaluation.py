import os
import time
import tensorflow as tf
import argparse
from src.madnet import MADNet
from src.preprocessing import StereoDatasetCreator
from src.losses_and_metrics import Bad3, EndPointError, SSIMLoss
from src.callbacks import TensorboardTestImagesCallback

parser = argparse.ArgumentParser(description='Script for evaluating MADNet')
parser.add_argument("--left_dir", help='Path to left images folder', required=True)
parser.add_argument("--right_dir", help='Path to right images folder', required=True)
parser.add_argument("--disp_dir", help='Path to left disparity maps folder. \n'
                                       'Note, groundtruth is not used here for training, '
                                       'its only used to extract the EPE and Bad3 metrics.', required=True)
parser.add_argument("--num_adapt", help='Number of modules to adapt, needs to be an integer from 0-6',
                    default=0, type=int, required=False)
parser.add_argument("--mad_mode", help='Module selection method for MAD adaptation. \n'
                                       'Options are "random" and "sequential".',
                    default="random", required=False)
parser.add_argument("--search_range", help='Maximum dispacement (ie. smallest disparity)',
                    default=2, type=int, required=False)
parser.add_argument("-o", "--output_dir",
                    help='Path for saving evaluation logs',
                    default=None, required=False)
parser.add_argument("--weights_path",
                    help='One of the following pretrained weights (will download automatically): '
                         '"synthetic", "kitti", "tf1_conversion_synthetic", "tf1_conversion_kitti"'
                         'or a path to pretrained MADNet weights file (for fine turning)',
                    default=None, required=False)
parser.add_argument("--lr", help="Learning rate (only needed if performing adaptation while inferencing).",
                    default=0.0001, type=float, required=False)
parser.add_argument("--height", help='Model image input height resolution', type=int, default=480)
parser.add_argument("--width", help='Model image input height resolution', type=int, default=640)
parser.add_argument("--batch_size", help='Batch size to use during training', type=int, default=1)
parser.add_argument("--log_tensorboard", help="Logs results to tensorboard events files.", action="store_true")
parser.add_argument("--repeat_data", help="Repeats the dataset. \n"
                                          "If you would like to perform evaluation on the dataset more than once. \n"
                                          "Make sure to set --steps with this so that it doesnt evaluate infinitely.",
                    action="store_true")
parser.add_argument("--steps", help='Number of steps to evaluate, set to None to inference all the data',
                    default=None, type=int, required=False)
parser.add_argument("--test_steps", help='Frequency of steps to log tensorboard data.',
                    default=5, type=int, required=False)
parser.add_argument("--save_predictions", help="Saves the disparity predictions.", action="store_true")

args = parser.parse_args()


def main(args):
    if args.repeat_data and args.steps is None:
        raise ValueError("Please set steps when repeating the data to avoid an infinite loop. \n"
                         f"Provided args: repeat_data: {args.repeat_data}, steps: {args.steps}")
    if args.log_tensorboard and args.output_dir is None:
        raise ValueError("Please provide an output dir when logging to tensorboard. \n"
                         f"Provided args: log_tensorboard: {args.log_tensorboard}, output_dir: {args.output_dir}")
    if args.output_dir is not None:
        # Create output folder if it doesn't already exist
        os.makedirs(args.output_dir, exist_ok=True)
        log_dir = args.output_dir + "/logs"
    images_dir = None
    if args.save_predictions:
        images_dir = args.output_dir + "/prediction_images"
        os.makedirs(images_dir, exist_ok=True)

    # Initialise the model
    model = MADNet(
        input_shape=(args.height, args.width, 3),
        weights=args.weights_path,
        num_adapt_modules=args.num_adapt,
        mad_mode=args.mad_mode,
        search_range=args.search_range
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss=SSIMLoss(),
        metrics=[EndPointError(), Bad3()],
        run_eagerly=False
    )
    # Get eval data
    eval_dataset = StereoDatasetCreator(
        left_dir=args.left_dir,
        right_dir=args.right_dir,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        shuffle=False,
        disp_dir=args.disp_dir
    )
    eval_ds = eval_dataset()
    if args.repeat_data:
        eval_ds = eval_ds.repeat()
    # Add callbacks
    all_callbacks = []
    if args.log_tensorboard:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_steps_per_second=True,
            update_freq="batch"
        )
        all_callbacks.append(tensorboard_callback)
        tensorboard_images_callback = TensorboardTestImagesCallback(
            testing_data=eval_ds,
            test_steps=args.test_steps,
            pred_dir=images_dir
        )
        all_callbacks.append(tensorboard_images_callback)

    # Evaluate the dataset
    start_time = time.time()
    results_dict = model.evaluate(
        eval_ds,
        steps=args.steps,
        callbacks=all_callbacks,
        return_dict=True
    )
    evaluate_sec = time.time() - start_time
    print(f"Evaluation took: {evaluate_sec // 60} min, {evaluate_sec % 60} s")
    print(f"Evaluation total seconds: {evaluate_sec} s")
    try:
        print(f"FPS: {args.steps / evaluate_sec}")
    except:
        print("Specify steps to get FPS")

    print(f"Results: {results_dict}")


if __name__ == "__main__":
    main(args)