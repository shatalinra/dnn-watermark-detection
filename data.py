from pathlib import Path
import tensorflow as tf

def load_image(path):
    image_file = tf.io.read_file(str(path))
    image = tf.io.decode_png(image_file)
    image = tf.cast(image, dtype = tf.float32) / 255
    return image

def apply_watermark(original, x, y, watermark):
    rgb = watermark[:,:,:3]
    mask = watermark[:,:,3:4] > 0
    original_shape = original.get_shape().as_list()
    original_rows, original_cols = original_shape[0], original_shape[1]
    watermark_shape = watermark.get_shape().as_list()
    watermark_rows, watermark_cols = watermark_shape[0], watermark_shape[1]
    x_end = x + watermark_cols
    y_end = y + watermark_rows

    watermark_padded = tf.pad(rgb, [[y, original_rows - y_end], [x, original_cols - x_end], [0, 0]])
    mask_padded = tf.pad(mask, [[y, original_rows - y_end], [x, original_cols - x_end], [0,0]])
    result = tf.where(mask_padded, watermark_padded, original)
    return result

class ImageSequence:
    """Class for specifyings sequence of images in one folder"""
    def __init__(self, folder_path, start_id, end_id, digits=12, extension = "jpg"):
        self._folder_path = folder_path
        self._start_id = start_id
        self._end_id = end_id
        self._digits = digits
        self._extension = extension

    def paths(self):
        for id in range(self._start_id, self._end_id):
            image_path = Path(self._folder_path + str(id).zfill(self._digits) + "." + self._extension)

            # not all ids maybe included, skip them
            if not image_path.exists():
                continue

            yield image_path

    def images(self):
        for image_path in self.paths():
            image = load_image(image_path)

            # skip grayscale images
            if image.get_shape().as_list()[2] == 1:
                continue

            yield image


class WatermarkedImageDataset:
    """Dataset of MS COCO training images with and without added watermarks."""

    def __init__(self, original : ImageSequence, watermarks: ImageSequence, step, preprocessing, output_shape):
        self._original = original
        self._watermarks = watermarks
        self._step = step
        self._output_shape = output_shape
        self._preprocessing = preprocessing

    def data_generator(self):
        for original in self._original.images():
            original_shape = original.get_shape().as_list()
            original_rows, original_cols = original_shape[0], original_shape[1]

            original_out = original
            if self._preprocessing is not None:
                original_out = self._preprocessing(original)

            for watermark in self._watermarks.images():
                watermark_shape = watermark.get_shape().as_list()
                watermark_rows, watermark_cols = watermark_shape[0], watermark_shape[1]
                for x in range(0, original_cols, self._step):
                    # skip locations where watermark would go out of bonds
                    if x + watermark_cols >= original_cols:
                            break

                    for y in range(0, original_rows, self._step):
                        # skip locations where watermark would go out of bonds
                        if y + watermark_rows >= original_rows:
                            break

                        # right now I am lazy and this seems to be a good way to balance things
                        yield original_out, tf.constant(0)

                        image = apply_watermark(original, x, y, watermark)
                        watermarked_out = image
                        if self._preprocessing is not None:
                            watermarked_out = self._preprocessing(image)
                        yield watermarked_out, tf.constant(1)

    def data(self):
        """ Collect all data in tensors which allows fitting small dataset on GPU completely"""
        all_images = None
        all_labels = None

        for patches, labels in self.data_generator():
            if all_images is None:
                all_images = patches
            else:
                all_images = tf.concat([all_images, patches], 0)
            if all_labels is None:
                all_labels = labels
            else:
                all_labels = tf.concat([all_labels, labels], 0)

        return tf.resize(all_images, [-1, self._output_image_size, self._output_image_size, 3]), all_labels

    def dataset(self):
        """ Generate tf.data.Dataset which uses CPU for all its internal processing and allows creating large datasets"""

        data_spec = tf.TensorSpec(shape=self._output_shape, dtype=tf.float32)
        label_spec = tf.TensorSpec(shape=(), dtype=tf.int32)

        return tf.data.Dataset.from_generator(self.data_generator, output_signature = (data_spec, label_spec))