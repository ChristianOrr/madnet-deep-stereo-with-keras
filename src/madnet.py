import tensorflow as tf
import keras
from src.model_layers import WarpImageBlock, ModuleM, RefinementBlock

@keras.saving.register_keras_serializable(package="MADNet")
class MADNet(keras.Model):
    f"""
    Instantiates the MADNet architecture

    Reference:
        - [MADNet: Real-time self-adaptive deep stereo](
          https://arxiv.org/abs/1810.05424) (CVPR 2019)

    Args:
        input_shape: Optional shape tuple, to be specified if you would
            like to use a model with an input image resolution that is not
            (480, 640, 3).
            It should have exactly 3 inputs channels (480, 640, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
        weights: String, one of `None` (random initialization),
            or one of the pretrained weights,
            or the path to the weights file to be loaded.
        input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        num_adapt_modules: Integer, number of modules to perform adaptation on while inferencing.
            For standard inferencing, use num_adapt_modules=0,
            MAD is num_adapt_modules=2-5,
            Full backprop is num_adapt_modules=6.
            Note: This is for inferencing only, so doesnt affect training.
            If you would like to change the inferencing mode you will need to
            instantiate the model again with the new num_adapt_modules value.
        mad_mode: String, one of "random" or "sequential"
            This is only needed for MAD adaptation with num_adapt_modules in 1-5.
            "random", selects the modules to adapt randomly.
            "sequential", selects the modules to adapt sequentially. 
        search_range: maximum search displacement for the cost volume

    Returns:
        A `keras.Model` instance.
    """
    def __init__(
        self,
        search_range=2,
        **kwargs
    ):
        super(MADNet, self).__init__(**kwargs)
        self.search_range = search_range

    def build(self, input_shape):
        left_shape = input_shape["left_input"]
        right_shape = input_shape["right_input"]
        assert left_shape == right_shape, "Left and right image shapes must be the same."
        image_shape = left_shape
        batch_size, height, width, channels = image_shape

        # Train step layers
        self.warp_block = WarpImageBlock()

        # Normalization layers
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name="feature_norm1")
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name="feature_norm2")
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-6, name="feature_norm3")
        self.norm4 = keras.layers.LayerNormalization(epsilon=1e-6, name="feature_norm4")
        self.norm5 = keras.layers.LayerNormalization(epsilon=1e-6, name="feature_norm5")
        self.norm6 = keras.layers.LayerNormalization(epsilon=1e-6, name="feature_norm6")
        self.norm7 = keras.layers.LayerNormalization(epsilon=1e-6, name="feature_norm7")
        self.norm8 = keras.layers.LayerNormalization(epsilon=1e-6, name="feature_norm8")
        self.norm9 = keras.layers.LayerNormalization(epsilon=1e-6, name="feature_norm9")
        self.norm10 = keras.layers.LayerNormalization(epsilon=1e-6, name="feature_norm10")
        self.norm11 = keras.layers.LayerNormalization(epsilon=1e-6, name="feature_norm11")
        self.norm12 = keras.layers.LayerNormalization(epsilon=1e-6, name="feature_norm12")   

        # Initializing the layers
        self.layer_kwargs = {
            "kernel_size": (3, 3),
            "padding": "same",
            "activation": keras.layers.Activation(keras.activations.leaky_relu, dtype="float32", name="leaky_relu"),
            "use_bias": True,
            "kernel_initializer": "he_normal"
        }
        # Image feature pyramid (feature extractor)
        # F1
        self.conv1 = keras.layers.Conv2D(
            filters=16,
            strides=2,
            name="conv1",
            # input_shape=(batch_size, height, width, ),
            **self.layer_kwargs)
        self.conv2 = keras.layers.Conv2D(filters=16, strides=1, name="conv2", **self.layer_kwargs)
        # F2
        self.conv3 = keras.layers.Conv2D(filters=32, strides=2, name="conv3", **self.layer_kwargs)
        self.conv4 = keras.layers.Conv2D(filters=32, strides=1, name="conv4", **self.layer_kwargs)
        # F3
        self.conv5 = keras.layers.Conv2D(filters=64, strides=2, name="conv5", **self.layer_kwargs)
        self.conv6 = keras.layers.Conv2D(filters=64, strides=1, name="conv6", **self.layer_kwargs)
        # F4
        self.conv7 = keras.layers.Conv2D(filters=96, strides=2, name="conv7", **self.layer_kwargs)
        self.conv8 = keras.layers.Conv2D(filters=96, strides=1, name="conv8", **self.layer_kwargs)
        # F5
        self.conv9 = keras.layers.Conv2D(filters=128, strides=2, name="conv9", **self.layer_kwargs)
        self.conv10 = keras.layers.Conv2D(filters=128, strides=1, name="conv10", **self.layer_kwargs)
        # F6
        self.conv11 = keras.layers.Conv2D(filters=192, strides=2, name="conv11", **self.layer_kwargs)
        self.conv12 = keras.layers.Conv2D(filters=192, strides=1, name="conv12", **self.layer_kwargs)

        #############################SCALE 6#################################
        self.M6 = ModuleM(layer="6", search_range=self.search_range)
        ############################SCALE 5###################################
        self.M5 = ModuleM(layer="5", search_range=self.search_range)
        ############################SCALE 4###################################
        self.M4 = ModuleM(layer="4", search_range=self.search_range)
        ############################SCALE 3###################################
        self.M3 = ModuleM(layer="3", search_range=self.search_range)
        ############################SCALE 2###################################
        self.M2 = ModuleM(layer="2", search_range=self.search_range)
        ############################REFINEMENT################################
        self.refinement = RefinementBlock(out_height=height, out_width=width)

    @tf.function(jit_compile=True)
    def train_step(self, data):
        """
        This adds the following training features:
            1. Training without groundtruth disparity. (self-supervised training)
            2. Tensorboard summaries.
            3. Loss is reduced for batch sizes larger than 1.
        """
        # Left, right image inputs and groundtruth target disparity
        inputs = data
        left_input = inputs["left_input"]
        right_input = inputs["right_input"]

        with tf.GradientTape(persistent=False) as tape:
            # Forward pass
            final_disparity = self(inputs=inputs, training=True)
            # Calculate loss using compiled_loss (ensures correct handling of loss objects)
            if "disp_map" not in inputs.keys():
                # Self-supervised: use image reprojection (left image vs warped right)
                warped_left = self.warp_block([right_input, final_disparity])
                y_true = left_input
                y_pred = warped_left
            else:
                # Supervised training: y_true is the ground-truth disparity map
                y_true = inputs["disp_map"]
                y_pred = final_disparity

            # compute_loss returns the (possibly unreduced) loss; add regularization losses manually
            loss = self.compute_loss(x=inputs, y=y_true, y_pred=y_pred)
            # Add any regularization losses that may be present on the model
            if self.losses:
                reg_loss = keras.ops.add_n(self.losses)
                if reg_loss is not None:
                    loss = loss + reg_loss
            # Perform reduction on the loss for backprop
            batch_size = keras.ops.shape(left_input)[0]
            reduced_loss = loss / keras.ops.cast(batch_size, dtype="float32")

    # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(reduced_loss, trainable_vars)   

        # Run backwards pass.
        # filter out None grads and apply
        grads_and_vars = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
        if grads_and_vars:
            grads, vars_ = zip(*grads_and_vars)
            self.optimizer.apply_gradients(zip(grads, vars_))

        # Update metrics manually (replacement for deprecated compiled_metrics)
        for metric in self.metrics:
            metric.update_state(y_true, y_pred)

        return {m.name: m.result() for m in self.metrics}

    @tf.function(jit_compile=True)
    def test_step(self, data):
        inputs = data
        # Use the model's predict_step to get predictions for this test batch
        y_pred = self.predict_step(data)

        # Compute and update loss-related state if needed
        y_true = inputs.get("disp_map", None)
        # If a loss is configured, compute it to ensure any stateful loss metrics are updated
        if y_true is not None:
            _ = self.compute_loss(x=inputs, y=y_true, y_pred=y_pred)

        for metric in self.metrics:
            metric.update_state(y_true, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=None):
        # left and right image inputs are set to the same resolution
        left_input = inputs["left_input"]
        right_input = inputs["right_input"]

        #######################PYRAMID FEATURES###############################
        # Left image feature pyramid (feature extractor)
        # F1
        left_input = self.norm1(left_input)
        left_pyramid = self.conv1(left_input)
        left_pyramid = self.norm2(left_pyramid)
        left_F1 = self.conv2(left_pyramid)
        # F2
        left_F1 = self.norm3(left_F1)
        left_pyramid = self.conv3(left_F1)
        left_pyramid = self.norm4(left_pyramid)
        left_F2 = self.conv4(left_pyramid)
        # F3
        left_F2 = self.norm5(left_F2)
        left_pyramid = self.conv5(left_F2)
        left_pyramid = self.norm6(left_pyramid)
        left_F3 = self.conv6(left_pyramid)
        # F4
        left_F3 = self.norm7(left_F3)
        left_pyramid = self.conv7(left_F3)
        left_pyramid = self.norm8(left_pyramid)
        left_F4 = self.conv8(left_pyramid)
        # F5
        left_F4 = self.norm9(left_F4)
        left_pyramid = self.conv9(left_F4)
        left_pyramid = self.norm10(left_pyramid)
        left_F5 = self.conv10(left_pyramid)
        # F6
        left_F5 = self.norm11(left_F5)
        left_pyramid = self.conv11(left_F5)
        left_pyramid = self.norm12(left_pyramid)
        left_F6 = self.conv12(left_pyramid)

        # Right image feature pyramid (feature extractor)
        # F1
        right_input = self.norm1(right_input)
        right_pyramid = self.conv1(right_input)
        right_pyramid = self.norm2(right_pyramid)
        right_F1 = self.conv2(right_pyramid)
        # F2
        right_F1 = self.norm3(right_F1)
        right_pyramid = self.conv3(right_F1)
        right_pyramid = self.norm4(right_pyramid)
        right_F2 = self.conv4(right_pyramid)
        # F3
        right_F2 = self.norm5(right_F2)
        right_pyramid = self.conv5(right_F2)
        right_pyramid = self.norm6(right_pyramid)
        right_F3 = self.conv6(right_pyramid)
        # F4
        right_F3 = self.norm7(right_F3)
        right_pyramid = self.conv7(right_F3)
        right_pyramid = self.norm8(right_pyramid)
        right_F4 = self.conv8(right_pyramid)
        # F5
        right_F4 = self.norm9(right_F4)
        right_pyramid = self.conv9(right_F4)
        right_pyramid = self.norm10(right_pyramid)
        right_F5 = self.conv10(right_pyramid)
        # F6
        right_F5 = self.norm11(right_F5)
        right_pyramid = self.conv11(right_F5)
        right_pyramid = self.norm12(right_pyramid)
        right_F6 = self.conv12(right_pyramid)

        #############################SCALE 6#################################
        D6 = self.M6({"left": left_F6, "right": right_F6})
        ############################SCALE 5###################################
        D5 = self.M5({"left": left_F5, "right": right_F5, "prev_disp": D6})
        ############################SCALE 4###################################
        D4 = self.M4({"left": left_F4, "right": right_F4, "prev_disp": D5})
        ############################SCALE 3###################################
        D3 = self.M3({"left": left_F3, "right": right_F3, "prev_disp": D4})
        ############################SCALE 2###################################
        D2 = self.M2({"left": left_F2, "right": right_F2, "prev_disp": D3})
        ############################REFINEMENT################################
        final_disparity = self.refinement([left_F2, D2])

        return final_disparity
