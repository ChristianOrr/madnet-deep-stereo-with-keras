import keras
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
    colors = color_map(keras.ops.arange(256))[:, :3]
    colors = keras.ops.convert_to_tensor(colors, dtype="float32")
    value = keras.ops.take(colors, indices, axis=0)
    return value


# https://github.com/philferriere/tfoptflow/blob/bdc7a72e78008d1cd6db46e4667dffc2bab1fe9e/tfoptflow/core_costvol.py
@keras.saving.register_keras_serializable(package="MADNet")
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

@keras.saving.register_keras_serializable(package="MADNet")
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

    def call(self, inputs):
        imgs, coords = inputs
        # compute shapes dynamically from tensors so batch size can vary
        coords_shape = keras.ops.shape(coords)
        imgs_shape = keras.ops.shape(imgs)

        batch_size = coords_shape[0]
        tgt_h = coords_shape[1]
        tgt_w = coords_shape[2]
        in_h = imgs_shape[1]
        in_w = imgs_shape[2]

        coords_x, coords_y = keras.ops.split(coords, 2, axis=3)
        coords_x = keras.ops.cast(coords_x, 'float32')
        coords_y = keras.ops.cast(coords_y, 'float32')

        x0 = keras.ops.floor(coords_x)
        x1 = x0 + 1
        y0 = keras.ops.floor(coords_y)
        y1 = y0 + 1

        y_max = keras.ops.cast(in_h - 1, 'float32')
        x_max = keras.ops.cast(in_w - 1, 'float32')
        zero = keras.ops.zeros([1], dtype='float32')

        wt_x0 = x1 - coords_x
        wt_x1 = coords_x - x0
        wt_y0 = y1 - coords_y
        wt_y1 = coords_y - y0

        x0_safe = keras.ops.clip(x0, zero[0], x_max)
        y0_safe = keras.ops.clip(y0, zero[0], y_max)
        x1_safe = keras.ops.clip(x1, zero[0], x_max)
        y1_safe = keras.ops.clip(y1, zero[0], y_max)

        # indices in the flattened image: base + y * width + x
        dim2 = keras.ops.cast(in_w, 'float32')
        dim1 = keras.ops.cast(in_w * in_h, 'float32')

        # base for each batch element repeated for each target pixel
        base_arange = keras.ops.cast(keras.ops.arange(batch_size), 'float32') * dim1
        # number of repeats per batch element
        n_repeats = tgt_h * tgt_w
        base = keras.ops.reshape(
            self._repeat(base_arange, n_repeats),
            [batch_size, tgt_h, tgt_w, 1]
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

@keras.saving.register_keras_serializable(package="MADNet")
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

        # Create 1D ranges for width (x) and height (y) using the Keras backend
        x_coords = keras.ops.arange(max_width, dtype="float32")
        y_coords = keras.ops.arange(max_height, dtype="float32")

        # Use meshgrid to create 2D coordinate grids from the 1D ranges.
        # 'ij' indexing ensures the output shape matches the (height, width) convention.
        y_grid, x_grid = keras.ops.meshgrid(y_coords, x_coords, indexing='ij')

        # Stack the x and y grids along a new last axis to get shape (height, width, 2)
        # The order [x_grid, y_grid] matches your original loop's [j, i] assignment.
        pixel_coords = keras.ops.stack([x_grid, y_grid], axis=-1)

        # Add the batch dimension to get the final shape (1, height, width, 2)
        pixel_coords = keras.ops.expand_dims(pixel_coords, axis=0)

        # pixel_coords = keras.ops.convert_to_tensor(pixel_coords, "float32")
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

@keras.saving.register_keras_serializable(package="MADNet")
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

@keras.saving.register_keras_serializable(package="MADNet")
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

@keras.saving.register_keras_serializable(package="MADNet")
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
