import os
import tensorflow as tf
import keras


class StereoDatasetCreator():
    """
    Takes paths to left and right stereo image directories
    and creates a tf.data.Dataset that returns a batch of left
    and right images. (Optional) Returns the disparities as a target
    using the disparities directories.
    Init Args:
        left_dir: path to left images folder
        right_dir: path to right images folder
        batch_size: desired batch size 
        height: desired height of the image (will be reshaped to this height if necessary)
        width: desired width of the image (will be reshaped to this width if necessary)
        shuffle: True/False
        (Optional) disp_dir: path to disparity maps folder
    Returns:
        object that can be called to return a tf.data.Dataset
        dataset will return values of the form: 
            {'left_input': (batch, height, width, 3), 'right_input': (batch, height, width, 3)},
            (Optional) (batch, height, width, 1) else None

    This can prepare MADNet data for training/evaluation and prediction
    """
    def __init__(self, left_dir, right_dir, height, width, batch_size=1, shuffle=False, disp_dir=None, augment=False):
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.disp_dir = disp_dir
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.shuffle = shuffle
        self.augment = augment

        self.left_names = keras.ops.array(
            sorted([str(name) for name in os.listdir(left_dir) if os.path.isfile(f"{self.left_dir}/{name}")]
            )
        )
        self.right_names = keras.ops.array(
            sorted([str(name) for name in os.listdir(right_dir) if os.path.isfile(f"{self.right_dir}/{name}")])
        )
        if self.disp_dir is not None:
            self.disp_names = keras.ops.array(
                sorted([str(name) for name in os.listdir(disp_dir) if os.path.isfile(f"{self.disp_dir}/{name}")])
            )

        # Check that there is a left image for every right image
        self.num_left = len(self.left_names)
        self.num_right = len(self.right_names)
        if self.num_left != self.num_right:
            raise ValueError(f"Number of right and left images do not match. "
                             f"Left number: {self.num_left}. Right number: {self.num_right}")
        if self.disp_dir is not None:
            self.num_disp = len(self.disp_names)
            if self.num_disp != self.num_left:
                raise ValueError(f"Number of disparity and left/right images do not match. "
                                 f"Disparity number: {self.num_disp}. "
                                 f"Left number: {self.num_left}. "
                                 f"Right number: {self.num_right}.")

    def _get_image(self, path):
        """
        Get a single image helper function
        Converts image to float32, normalises values to 0-1
        and resizes to the desired shape
        Args:
            path to image (will be in Tensor format, since its called in a graph)
        Return:
            Tensor in the shape (height, width, 3)
        """
        # Using tf.io.read_file since it can take a tensor as input
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, dtype="float32", expand_animations=False)
        image = keras.ops.image.resize(image, [self.height, self.width], interpolation="bilinear")
        if self.augment:
            image = tf.image.random_hue(image, 0.08)
            image = tf.image.random_saturation(image, 0.6, 1.6)
            image = tf.image.random_contrast(image, 0.7, 1.3)
        return image

    def _read_pfm_tf(self, file_path):
        """
        Loads a PFM file as a Keras/TensorFlow tensor.
        This is a graph-compatible replacement for the original NumPy-based readPFM.
        
        Args:
            file_path: A scalar string tensor containing the path to the file.
            
        Returns:
            A tensor with shape (height, width, channels).
        """
        # 1. Read the entire file content into a single tensor
        raw_content = tf.io.read_file(file_path)

        # 2. Split the header from the binary data. The header is 3 lines.
        # We split into 4 parts: header, dims, scale, and the rest (binary data)
        parts = tf.strings.split(raw_content, sep='\n', maxsplit=3)
        header_line = parts[0]
        dims_line = parts[1]
        scale_line = parts[2]
        binary_data = parts[3]

        # 3. Parse the header information using TensorFlow string ops
        # 'PF' = color (3 channels), 'Pf' = grayscale (1 channel)
        is_color = tf.equal(header_line, 'PF')

        # Parse width and height
        dims = tf.strings.to_number(tf.strings.split(dims_line, ' '), out_type=tf.int32)
        width, height = dims[0], dims[1]
        
        # Parse scale factor to determine endianness
        scale = tf.strings.to_number(scale_line, out_type=tf.float32)
        is_little_endian = scale < 0.0

        # 4. Decode the raw binary data. We must use tf.cond because the
        # 'little_endian' parameter for decode_raw cannot be a tensor.
        def decode_little_endian():
            return tf.io.decode_raw(binary_data, tf.float32, little_endian=True)

        def decode_big_endian():
            return tf.io.decode_raw(binary_data, tf.float32, little_endian=False)

        data_vector = tf.cond(is_little_endian, decode_little_endian, decode_big_endian)

        # 5. Reshape the data vector into the correct image shape
        # We use tf.cond again to handle the conditional channel number
        def get_color_shape():
            return tf.stack([height, width, 3])

        def get_mono_shape():
            return tf.stack([height, width, 1])

        shape = tf.cond(is_color, get_color_shape, get_mono_shape)
        image = keras.ops.reshape(data_vector, shape)
        
        image = keras.ops.flip(image, axis=0)

        return image

    def _get_pfm(self, path):
        """
        Reads a single pfm disparity file and returns a disparity map.
        This version is fully graph-compatible and uses Keras/TensorFlow ops only.
        
        Args:
            path: A scalar string tensor path to the disparity file.
            
        Returns:
            A tensor disparity map with shape (height, width, 1).
        """
        disp_map = self._read_pfm_tf(path)

        disp_map = keras.ops.where(keras.ops.isinf(disp_map), 0.0, disp_map)

        # Replace the Python `if` with a graph-compatible `keras.ops.cond`
        # Replace: if disp_map.mean() < 0: disp_map *= -1
        disp_map = keras.ops.cond(
            keras.ops.mean(disp_map) < 0.0,
            lambda: disp_map * -1.0,  # op to run if true
            lambda: disp_map          # op to run if false
        )

        # This operation was already using Keras, so it remains the same.
        disp_map = keras.ops.image.resize(
            disp_map, [self.height, self.width], interpolation="nearest"
        )
        
        return disp_map

    def _get_disp(self, disp_name):
        """
        Reads a disparity file (.pfm or .png) and returns a processed tensor map.
        This version is fully graph-compatible and uses Keras/TensorFlow ops only.
        
        Args:
            disp_name: A scalar string tensor, name of the disparity file.
            
        Returns:
            A tensor disparity map in the format [height, width, 1].
        """
        # Create the full file path using graph-compatible string operations
        disp_path = tf.strings.join([self.disp_dir, "/", disp_name])

        # Check the file extension using tensor-based operations
        # Use regex_full_match because `endswith` is a python string method
        # and not available on symbolic tensors.
        lower_name = tf.strings.lower(disp_name)
        is_pfm = tf.strings.regex_full_match(lower_name, r'.*\.pfm')
        is_png = tf.strings.regex_full_match(lower_name, r'.*\.png')

        # Assert that the file type is supported. This is the graph-compatible
        # way to raise an error if the condition is not met.
        tf.Assert(
            tf.logical_or(is_pfm, is_png),
            ["Unsupported disparity file detected. Only .pfm and .png are supported. Got:", disp_name]
        )

        def read_and_process_png():
            """Helper function to process PNG files."""
            disp_bytes = tf.io.read_file(disp_path)
            # Using uint16 for higher precision, channels=1 for grayscale
            disp_map = tf.io.decode_png(disp_bytes, dtype="uint16", channels=1)
            disp_map = keras.ops.cast(disp_map, dtype="float32")
            disp_map = disp_map / 256.0
            # Using nearest neighbor interpolation for sparse groundtruth disparities
            return keras.ops.image.resize(disp_map, [self.height, self.width], interpolation="nearest")

        # Use tf.cond to choose the correct processing path.
        # It takes a boolean tensor and two functions to execute.
        disp_map = tf.cond(
            is_pfm,
            # Function to run if `is_pfm` is True.
            # It calls the `_get_pfm` function we previously converted.
            true_fn=lambda: self._get_pfm(disp_path),
            # Function to run if `is_pfm` is False.
            false_fn=read_and_process_png
        )

        return disp_map

    def _process_single_batch(self, index):
        """
        Processes a single batch using index to find the files
        Args: 
            index: Tensor integer
        Returns:
            stereo input dictionary, target
        """
        left_name = self.left_names[index]
        right_name = self.right_names[index]
        left_image = self._get_image(f"{self.left_dir}/" + left_name)
        right_image = self._get_image(f"{self.right_dir}/" + right_name)

        disp_map = None  
        if self.disp_dir is not None:
            disp_name = self.disp_names[index]
            disp_map = self._get_disp(disp_name)
            # restore static shape information so Dataset.element_spec is known
            disp_map = tf.ensure_shape(disp_map, (self.height, self.width, 1))

            return {'left_input': left_image, 'right_input': right_image, "disp_map": disp_map}
        else:
            return {'left_input': left_image, 'right_input': right_image}

    def __call__(self):
        """
        Creates and returns a tensorflow data.Dataset
        The dataset is shuffled, batched and prefetched
        """
        indexes = list(range(self.num_left))
        indexes_ds = tf.data.Dataset.from_tensor_slices(indexes)
        if self.shuffle:
            indexes_ds = indexes_ds.shuffle(buffer_size=self.num_left, seed=101, reshuffle_each_iteration=False)

        ds = indexes_ds.map(self._process_single_batch)
        ds = ds.batch(batch_size=self.batch_size, drop_remainder=True)
        ds = ds.prefetch(buffer_size=10)
        return ds
    