import os
import tensorflow as tf
import argparse
from datetime import datetime
from src.madnet import MADNet
from src.preprocessing import StereoDatasetCreator
from src.losses_and_metrics import Bad3, EndPointError, ReconstructionLoss, SSIMLoss
import wandb
from wandb.keras import WandbCallback
from src.callbacks import WandBImagesCallback, TensorboardImagesCallback

wandb.login()

parser = argparse.ArgumentParser(description='Script for training MADNet and '
                                             'logging to Weights and Biases dashboard.')
parser.add_argument("--train_left_dir", help='path to left images folder', required=True)
parser.add_argument("--train_right_dir", help='path to right images folder', required=True)
parser.add_argument("--train_disp_dir", help='path to left disparity maps folder', default=None, required=False)
parser.add_argument("--val_left_dir", help='path to left images folder', default=None, required=False)
parser.add_argument("--val_right_dir", help='path to right images folder', default=None, required=False)
parser.add_argument("--val_disp_dir", help='path to left disparity maps folder', default=None, required=False)
parser.add_argument("--shuffle", help='shuffle training dataset', action="store_true", default=False)
parser.add_argument("--search_range", help='maximum displacement (ie. smallest disparity)',
                    default=2, type=int, required=False)
parser.add_argument("-o", "--output_dir",
                    help='path to folder for outputting tensorboard logs and saving model weights',
                    required=True)
parser.add_argument("--weights_path",
                    help='One of the following pretrained weights (will download automatically): '
                         '"synthetic", "kitti", "tf1_conversion_synthetic", "tf1_conversion_kitti"'
                         'or a path to pretrained MADNet weights file (for fine turning)',
                    default=None, required=False)
parser.add_argument("--lr", help="Initial value for learning rate.", default=0.0001, type=float, required=False)
parser.add_argument("--min_lr", help="Minimum learning rate cap.", default=0.0000001, type=float, required=False)
parser.add_argument("--decay", help="Exponential decay rate.", default=0.999, type=float, required=False)
parser.add_argument("--height", help='model image input height resolution', type=int, default=480)
parser.add_argument("--width", help='model image input height resolution', type=int, default=640)
parser.add_argument("--batch_size", help='batch size to use during training', type=int, default=1)
parser.add_argument("--num_epochs", help='number of training epochs', type=int, default=1000)
parser.add_argument("--epoch_steps", help='training steps per epoch', type=int, default=1000)
parser.add_argument("--save_freq", help='model saving frequency per steps', type=int, default=1000)
parser.add_argument("--epoch_evals", help='number of epochs per evaluation', type=int, default=1)
parser.add_argument("--dataset_name", help="Name of the dataset being trained on",
                    default="FlyingThings3D", required=False)
parser.add_argument("--log_tensorboard", help="Logs results to tensorboard events files.", action="store_true")
parser.add_argument("--use_checkpoints",
                    help="Saves the weights using the tensorflow checkpoints format.",
                    action="store_true")
parser.add_argument("--sweep", help="Creates new output sub-folders for each sweep.", action="store_true")
parser.add_argument("--augment", help="Performs augmentation on the left and right images.", action="store_true")
args = parser.parse_args()


def main(args):
    self_supervised = False
    if args.train_disp_dir is None:
        self_supervised = True
    if args.sweep:
        now = datetime.now()
        current_time = now.strftime("%Y%m%dT%H%MZ")
        args.output_dir = args.output_dir + f"/sweep-{current_time}"
    log_dir = args.output_dir + "/logs"
    save_extension = ".h5"
    if args.use_checkpoints:
        save_extension = ".ckpt"

    # Initialize wandb with your project
    run = wandb.init(project='madnet-keras',
                     sync_tensorboard=True,
                     config={
                         "learning_rate": args.lr,
                         "exponential_decay": args.decay,
                         "epochs": args.num_epochs,
                         "batch_size": args.batch_size,
                         "search_range": args.search_range,
                         "self_supervised_training": self_supervised,
                         "loss_function": "adam",
                         "architecture": "MADNet",
                         "dataset": args.dataset_name
                     })
    config = wandb.config

    perform_val = False
    if args.val_left_dir is not None and args.val_right_dir is not None and args.val_disp_dir is not None:
        perform_val = True
    # Create output folder if it doesn't already exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialise the model
    model = MADNet(
        input_shape=(args.height, args.width, 3),
        weights=args.weights_path,
        search_range=args.search_range
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    # If no train groundtruth is available, then the reprojection error
    # from warping is used to calculate the loss
    if self_supervised:
        model.compile(
            optimizer=optimizer,
            loss=SSIMLoss(),
            metrics=[EndPointError(), Bad3()],
            run_eagerly=True if perform_val else False
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=ReconstructionLoss(),
            metrics=[EndPointError(), Bad3()],
            run_eagerly=False
        )

    # Get dataset for training
    train_dataset = StereoDatasetCreator(
        left_dir=args.train_left_dir,
        right_dir=args.train_right_dir,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        shuffle=args.shuffle,
        disp_dir=args.train_disp_dir,
        augment=args.augment
    )
    train_ds = train_dataset().repeat()
    # Get datasets for training and callbacks
    train_callback_dataset = StereoDatasetCreator(
        left_dir=args.train_left_dir,
        right_dir=args.train_right_dir,
        batch_size=1,
        height=args.height,
        width=args.width,
        shuffle=args.shuffle,
        disp_dir=args.train_disp_dir,
        augment=args.augment
    )
    train_callback_ds = train_callback_dataset().repeat()
    val_ds = None
    if perform_val:
        val_dataset = StereoDatasetCreator(
            left_dir=args.val_left_dir,
            right_dir=args.val_right_dir,
            batch_size=1,
            height=args.height,
            width=args.width,
            shuffle=args.shuffle,
            disp_dir=args.val_disp_dir
        )
        val_ds = val_dataset().repeat()

    # Create callbacks
    def scheduler(epoch, lr):
        min_lr = args.min_lr
        if epoch > 100:
            # learning_rate * decay_rate ^ (global_step / decay_steps)
            lr = lr * args.decay ** (epoch // 100)
        lr = max(min_lr, lr)
        tf.summary.scalar('learning rate', data=lr, step=epoch)
        return lr
    schedule_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.output_dir + "/epoch-{epoch:04d}" + save_extension,
        save_freq=args.save_freq,
        save_weights_only=True,
        verbose=0
    )
    wandb_callback = WandbCallback(
        monitor="loss",
        mode="min",
        save_model=False,   # Keep False, Hangs for a long time and doesn't finish the run
        save_graph=False,   # Keep False, Crashes script
        save_weights_only=True  # Keep True, full model is very large, 300MB. With just weights its 45MB.
        )
    wandb_images_callback = WandBImagesCallback(
        training_data=train_callback_ds,
        validation_data=val_ds,
        val_epochs=args.epoch_evals
    )
    nan_callback = tf.keras.callbacks.TerminateOnNaN()
    all_callbacks = [
            save_callback,
            schedule_callback,
            wandb_callback,
            wandb_images_callback,
            nan_callback
        ]
    if args.log_tensorboard:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_steps_per_second=True,
            update_freq="batch"
        )
        all_callbacks.append(tensorboard_callback)
        tensorboard_images_callback = TensorboardImagesCallback(
            training_data=train_callback_ds,
            validation_data=val_ds,
            val_epochs=args.epoch_evals
        )
        all_callbacks.append(tensorboard_images_callback)

    # Fit the model
    history = model.fit(
        x=train_ds,
        epochs=args.num_epochs,
        verbose=1,
        steps_per_epoch=args.epoch_steps,
        callbacks=all_callbacks
    )
    run.finish()


if __name__ == "__main__":
    main(args)