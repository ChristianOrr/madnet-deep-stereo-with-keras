import random
import tensorflow as tf
import keras
from keras import backend
from keras.utils import get_file
from keras.utils import get_source_inputs
from keras import layers
import numpy as np
from matplotlib import cm


def colorize_img(value, vmin=None, vmax=None, cmap='jet'):
    """
    A utility function for TensorFlow that maps a grayscale image to
    a matplotlib colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1
    before mapping to a grayscale colormap.
    Arguments:
      - value: 4D Tensor of shape [batch_size, height, width, 1]
      - vmin: the minimum value of the range used for normalization. (Default: value minimum)
      - vmax: the maximum value of the range used for normalization. (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's 'get_cmap'.(Default: 'gray')
    
    Returns a 3D tensor of shape [batch_size, height, width, 3].
    """
    # Add a new dimension if the input tensor has rank 3
    if len(value.shape) == 3:
        value = value[..., None]

    # normalize
    vmin = keras.ops.min(value) if vmin is None else vmin
    vmax = keras.ops.max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # quantize
    indices = keras.ops.cast(keras.ops.round(value[:, :, :, 0]*255), dtype="int32")

    # gather
    color_map = cm.get_cmap(cmap)
    colors = color_map(np.arange(256))[:, :3]
    colors = keras.ops.convert_to_tensor(colors, dtype="float32")
    value = keras.ops.take(colors, indices, axis=0)
    return value


# https://github.com/philferriere/tfoptflow/blob/bdc7a72e78008d1cd6db46e4667dffc2bab1fe9e/tfoptflow/core_costvol.py
class CostVolumeBlock(keras.layers.Layer):
    """Build cost volume for associating a pixel from the
    left image with its corresponding pixels in the right image.
    Args:
        c1: Level of the feature pyramid of the left image
        warp: Warped level of the feature pyramid of the right image
        search_range: Search range (maximum displacement)
    """
    def __init__(self, search_range):
        super(CostVolumeBlock, self).__init__()
        self.search_range = search_range

    def build(self, input_shape):
        c1_shape, warp_shape = input_shape
        self.width = c1_shape[2]

    def call(self, inputs):
        c1, warp = inputs
        padded_lvl = keras.ops.pad(warp, [[0, 0], [0, 0], [self.search_range, self.search_range], [0, 0]])
        max_offset = self.search_range * 2 + 1

        cost_vol = []
        for i in range(0, max_offset):
            slice = padded_lvl[:, :, i:self.width+i, :]
            cost = keras.ops.mean(c1 * slice, axis=3, keepdims=True)
            cost_vol.append(cost)

        cost_vol = keras.ops.concatenate(cost_vol, axis=3)
        cost_curve = keras.ops.concatenate([c1, cost_vol], axis=3)

        return cost_curve


class BilinearSampler(keras.layers.Layer):
    """
    Construct a new image by bilinear sampling from the input image.
    Points falling outside the source image boundary have value 0.
    Args:
        imgs: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t,width_t, 2].
                height_t/width_t correspond to the dimensions of the output image
                (don't need to be the same as height_s/width_s). The two channels
                correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, height_t, width_t, channels]
    """
    def __init__(self):
        super(BilinearSampler, self).__init__()

    def _repeat(self, x, n_repeats):
        rep = keras.ops.transpose(
            keras.ops.expand_dims(keras.ops.ones(shape=keras.ops.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = keras.ops.cast(rep, 'float32')
        x = keras.ops.matmul(keras.ops.reshape(x, (-1, 1)), rep)
        return keras.ops.reshape(x, [-1])

    def build(self, input_shape):
        self.inp_size, self.coord_size = input_shape

    def call(self, inputs):
        imgs, coords = inputs
        coords_x, coords_y = keras.ops.split(coords, 2, axis=3)
        out_size = [self.coord_size[0], self.coord_size[1], self.coord_size[2], self.inp_size[3]]

        coords_x = keras.ops.cast(coords_x, 'float32')
        coords_y = keras.ops.cast(coords_y, 'float32')

        x0 = keras.ops.floor(coords_x)
        x1 = x0 + 1
        y0 = keras.ops.floor(coords_y)
        y1 = y0 + 1

        y_max = keras.ops.cast(self.inp_size[1] - 1, 'float32')
        x_max = keras.ops.cast(self.inp_size[2] - 1, 'float32')
        zero = keras.ops.zeros([1], dtype='float32')

        wt_x0 = x1 - coords_x
        wt_x1 = coords_x - x0
        wt_y0 = y1 - coords_y
        wt_y1 = coords_y - y0

        x0_safe = keras.ops.clip(x0, zero[0], x_max)
        y0_safe = keras.ops.clip(y0, zero[0], y_max)
        x1_safe = keras.ops.clip(x1, zero[0], x_max)
        y1_safe = keras.ops.clip(y1, zero[0], y_max)

        ## indices in the flat image to sample from
        dim2 = keras.ops.cast(self.inp_size[2], 'float32')
        dim1 = keras.ops.cast(self.inp_size[2] * self.inp_size[1], 'float32')
        base = keras.ops.reshape(
            self._repeat(keras.ops.cast(keras.ops.arange(self.coord_size[0]), 'float32') * dim1, self.coord_size[1] * self.coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1]
        )

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = x0_safe + base_y0
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        ## sample from imgs
        im00 = keras.ops.take(imgs, keras.ops.cast(idx00, 'int32'))
        im01 = keras.ops.take(imgs, keras.ops.cast(idx01, 'int32'))
        im10 = keras.ops.take(imgs, keras.ops.cast(idx10, 'int32'))
        im11 = keras.ops.take(imgs, keras.ops.cast(idx11, 'int32'))

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = keras.ops.add(
            keras.ops.add(w00 * im00, w01 * im01),
            keras.ops.add(w10 * im10, w11 * im11)
        )

        return output

class WarpImageBlock(keras.layers.Layer):
    """
    Given an image and a flow generate the warped image,
    for stereo img is the right image, flow is the disparity aligned with left.
    img: image that needs to be warped
    flow: Generic optical flow or disparity
    """

    def __init__(self):
        super(WarpImageBlock, self).__init__()
        self.bilinear_sampler = BilinearSampler()

    def build_coords(self, immy):
        max_height = 2048
        max_width = 2048
        pixel_coords = np.ones((1, max_height, max_width, 2))

        # build pixel coordinates and their disparity
        for i in range(0, max_height):
            for j in range(0, max_width):
                pixel_coords[0][i][j][0] = j
                pixel_coords[0][i][j][1] = i

        pixel_coords = keras.ops.convert_to_tensor(pixel_coords, "float32")
        real_height = keras.ops.shape(immy)[1]
        real_width = keras.ops.shape(immy)[2]
        real_pixel_coord = pixel_coords[:, 0:real_height, 0:real_width, :]
        immy = keras.ops.concatenate([immy, keras.ops.zeros_like(immy)], axis=-1)
        output = real_pixel_coord - immy

        return output

    def call(self, inputs):
        img, flow = inputs
        coords = self.build_coords(flow)
        warped = self.bilinear_sampler([img, coords])
        return warped


class RefinementBlock(keras.layers.Layer):
    """
    Final Layer in MADNet.
    Calculates the reprojection loss if training=True.
    Args:
        input: left_F2 tensor
        disp: D2 disparity from M2 module
        final_left: full resolution RGB left image
        final_right: full resolution RGB right image
    Returns:
        Full resolution disparity in float32 normalized 0-1
    """
    def __init__(self, out_height, out_width):
        super(RefinementBlock, self).__init__()
        self.out_height = out_height
        self.out_width = out_width

    def build(self, input_shape):
        layer_kwargs = {
            "kernel_size": (3, 3),
            "padding": "same",
            "activation": keras.layers.Activation(keras.activations.leaky_relu, dtype="float32", name="leaky_relu"),
            "use_bias": True,
            "kernel_initializer": "he_normal"
        }
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name="context_norm1")
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name="context_norm2")
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-6, name="context_norm3")
        self.norm4 = keras.layers.LayerNormalization(epsilon=1e-6, name="context_norm4")
        self.norm5 = keras.layers.LayerNormalization(epsilon=1e-6, name="context_norm5")
        self.norm6 = keras.layers.LayerNormalization(epsilon=1e-6, name="context_norm6")
        self.norm7 = keras.layers.LayerNormalization(epsilon=1e-6, name="context_norm7")
        self.context1 = keras.layers.Conv2D(filters=128, dilation_rate=1, name="context1", **layer_kwargs)
        self.context2 = keras.layers.Conv2D(filters=128, dilation_rate=2, name="context2", **layer_kwargs)
        self.context3 = keras.layers.Conv2D(filters=128, dilation_rate=4, name="context3", **layer_kwargs)
        self.context4 = keras.layers.Conv2D(filters=96, dilation_rate=8, name="context4", **layer_kwargs)
        self.context5 = keras.layers.Conv2D(filters=64, dilation_rate=16, name="context5", **layer_kwargs)
        self.context6 = keras.layers.Conv2D(filters=32, dilation_rate=1, name="context6", **layer_kwargs)
        self.context7 = keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            dilation_rate=1,
            padding="same",
            activation="linear",
            use_bias=True,
            kernel_initializer="he_normal",
            name="context7"
        )

    def call(self, inputs):
        input, disp = inputs
        volume = keras.ops.concatenate([input, disp], axis=-1)
        x = self.norm1(volume)
        x = self.context1(x)
        x = self.norm2(x)
        x = self.context2(x)
        x = self.norm3(x)
        x = self.context3(x)
        x = self.norm4(x)
        x = self.context4(x)
        x = self.norm5(x)
        x = self.context5(x)
        x = self.norm6(x)
        x = self.context6(x)
        x = self.norm7(x)
        x = self.context7(x)

        context_disp = keras.ops.add(disp, x)
        final_disparity = keras.ops.image.resize(
            images=context_disp,
            size=(self.out_height, self.out_width),
            interpolation='bilinear'
        )
        return final_disparity


class StereoEstimatorBlock(keras.layers.Layer):
    """
    This is the stereo estimation network at resolution n.
    It uses the costs (from the pixel difference between the warped right image 
    and the left image) combined with the upsampled disparity from the previous
    layer (when the layer is not the last layer).

    The output is predicted disparity for the network at resolution n.
    """
    def __init__(self, name):
        super(StereoEstimatorBlock, self).__init__()
        self.name = name

    def build(self, input_shape):
        layer_kwargs = {
            "kernel_size": (3, 3),
            "strides": 1,
            "padding": "same",
            "activation": keras.layers.Activation(keras.activations.leaky_relu, dtype="float32", name="leaky_relu"),
            "use_bias": True,
            "kernel_initializer": "he_normal"
        }
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_norm1")
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_norm2")
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_norm3")
        self.norm4 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_norm4")
        self.norm5 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_norm5")
        self.norm6 = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_norm6")
        self.disp1 = keras.layers.Conv2D(filters=128, name=f"{self.name}_disp1", **layer_kwargs)
        self.disp2 = keras.layers.Conv2D(filters=128, name=f"{self.name}_disp2", **layer_kwargs)
        self.disp3 = keras.layers.Conv2D(filters=96, name=f"{self.name}_disp3", **layer_kwargs)
        self.disp4 = keras.layers.Conv2D(filters=64, name=f"{self.name}_disp4", **layer_kwargs)
        self.disp5 = keras.layers.Conv2D(filters=32, name=f"{self.name}_disp5", **layer_kwargs)
        self.disp6 = keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="linear",
            use_bias=True,
            kernel_initializer="he_normal",
            name=f"{self.name}_disp6"
        )

    def call(self, inputs={"costs": None, "upsampled_disp": None}):
        if "upsampled_disp" in inputs.keys():
            volume = keras.ops.concatenate([inputs["costs"], inputs["upsampled_disp"]], axis=-1)
        else:
            volume = inputs["costs"]

        x = self.norm1(volume)
        x = self.disp1(x)
        x = self.norm2(x)
        x = self.disp2(x)
        x = self.norm3(x)
        x = self.disp3(x)
        x = self.norm4(x)
        x = self.disp4(x)
        x = self.norm5(x)
        x = self.disp5(x)
        x = self.norm6(x)
        x = self.disp6(x)
        return x


class ModuleM(keras.layers.Layer):
    """
    Module MX is a sub-module of MADNet, which can be trained individually for 
    online adaptation using the MAD (Modular ADaptaion) method.
    """
    def __init__(self, layer, search_range):
        super(ModuleM, self).__init__()
        self.warp_image_block = WarpImageBlock()
        self.layer = layer
        self.search_range = search_range
        self.cost_volume_block = CostVolumeBlock(search_range=search_range)
        self.stereo_estimator_block = StereoEstimatorBlock(name=f"volume_filtering_{self.layer}")

    def build(self, input_shape):
        self.mod_height, self.mod_width = input_shape["left"][1], input_shape["left"][2]

    def call(self, inputs={"left": None, "right": None, "prev_disp": None}):
        # Check if layer is the bottom of the pyramid
        if "prev_disp" in inputs.keys():
            # Upsample disparity from previous layer
            upsampled_disp = keras.ops.image.resize(
                images=inputs["prev_disp"],
                size=(self.mod_height, self.mod_width),
                interpolation='bilinear'
            )
            # Warp the right image into the left using upsampled disparity
            warped_left = self.warp_image_block([inputs["right"], upsampled_disp])
        else:
            # No previous disparity exits, so use right image instead of warped left
            warped_left = inputs["right"]

        costs = self.cost_volume_block([inputs["left"], warped_left])

        if "prev_disp" in inputs.keys():
            # Get the disparity using cost volume between left and warped left images
            module_disparity = self.stereo_estimator_block({"costs": costs, "upsampled_disp": upsampled_disp})
        else:
            module_disparity = self.stereo_estimator_block({"costs": costs})

        return module_disparity


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

    @tf.function
    def train_step(self, data):
        """
        This adds the following training features:
            1. Training without groundtruth disparity. (self-supervised training)
            2. Tensorboard summaries.
            3. Loss is reduced for batch sizes larger than 1.
        """
        # Left, right image inputs and groundtruth target disparity
        inputs, sample_weight = data
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

            # compiled_loss returns the (possibly unreduced) loss; include regularization losses
            loss = self.compiled_loss(y_true, y_pred, sample_weight, regularization_losses=self.losses)
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

        # Update compiled metrics; this handles both supervised and self-supervised cases
        self.compiled_metrics.update_state(y_true, y_pred, sample_weight)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        inputs, sample_weight = data
        y_pred = self.predict_step(self, data)
        # Updates stateful loss metrics.
        for metric in self.metrics:
            metric.update_state(inputs["disp_map"], y_pred, sample_weight)
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
