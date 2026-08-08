import os
import tensorflow as tf
import keras
from src.madnet import MADNet
from src.preprocessing import StereoDatasetCreator
from src.losses_and_metrics import Bad3, EndPointError, ReconstructionLoss, SSIMLoss
from src.callbacks import TensorboardImagesCallback
from keras.src.utils import file_utils


def run_train(args):
    perform_val = False
    if args.val_left_dir is not None and args.val_right_dir is not None and args.val_disp_dir is not None:
        perform_val = True
    # Create output folder if it doesn't already exist
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = args.output_dir + "/logs"
    save_extension = ".keras"

    model = MADNet(
        name="mad_net",
        search_range=args.search_range
    )
    # Force variable creation via a dummy forward pass (preferred over model.build())
    left = tf.zeros((args.batch_size, args.height, args.width, 3), dtype=tf.float32)
    right = tf.zeros_like(left)
    _ = model({"left_input": left, "right_input": right}, training=False)
    pretrained_weights = {"synthetic", "kitti", "tf1_conversion_synthetic", "tf1_conversion_kitti"}
    
    if args.weights_path is None:
        pass
    elif not (args.weights_path in pretrained_weights or file_utils.exists(args.weights_path)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), "
            "`synthetic` or `kitti`, "
            "or the path to the weights file to be loaded."
        )
    elif args.weights_path in pretrained_weights:
        pretrained_models_url = "https://huggingface.co/ChristianOrr/madnet_keras/resolve/main/"
        model_name = "madnet_" + args.weights_path + ".h5"
        weight_path = pretrained_models_url + args.weights_path + ".h5"
        weights_path = keras.utils.get_file(model_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path, by_name=True)        

    elif args.weights_path is not None:
        model.load_weights(args.weights_path)  

    optimizer = keras.optimizers.AdamW(learning_rate=args.lr)
    # If no train groundtruth is available, then the reprojection error
    # from warping is used to calculate the loss
    if args.train_disp_dir is None:
        model.compile(
            optimizer=optimizer,
            loss=SSIMLoss(),
            metrics=[EndPointError(), Bad3()],
            run_eagerly=False
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=ReconstructionLoss(),
            metrics=[EndPointError(), Bad3()],
            run_eagerly=False
        )
        
    # Get training data
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
        if epoch > 10:
            # learning_rate * decay_rate ^ (global_step / decay_steps)
            lr = lr * args.decay ** (epoch // 10)
        lr = max(min_lr, lr)
        tf.summary.scalar('learning rate', data=lr, step=epoch)
        return lr
    schedule_callback = keras.callbacks.LearningRateScheduler(scheduler)
    save_callback = keras.callbacks.ModelCheckpoint(
        filepath=args.output_dir + "/epoch-{epoch:04d}" + save_extension,
        save_freq=args.save_freq,
        save_weights_only=False,
        verbose=0
    )
    all_callbacks = [
            save_callback,
            schedule_callback
        ]
    if args.log_tensorboard:
        tensorboard_callback = keras.callbacks.TensorBoard(
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
    return history