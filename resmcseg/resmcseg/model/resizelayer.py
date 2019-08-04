from keras.utils import conv_utils
from keras.backend.common import normalize_data_format
from keras.engine import InputSpec
from keras.engine import Layer
import keras
import tensorflow as tf

class ResizeLayer(Layer):
    """ResizeLayer class: Resize Images to a specified size

    # Arguments
        output_size: Size of output layer width and height
        data_format: A string,
               one of `channels_last` (default) or `channels_first`.
              The ordering of the dimensions in the inputs.
               `channels_last` corresponds to inputs with shape
               `(batch, height, width, channels)` while `channels_first`
                  corresponds to inputs with shape
                `(batch, channels, height, width)`.
                 It defaults to the `image_data_format` value found in your
                  Keras config file at `~/.keras/keras.json`.
                  If you never set it, then it will be "channels_last".

    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`

    # Output shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, pooled_rows, pooled_cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, pooled_rows, pooled_cols)`
    """
    def __init__(self, output_dim=(1, 1), data_format=None, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        data_format = normalize_data_format(data_format)
        self.output_dim = conv_utils.normalize_tuple(output_dim, 2, 'output_dim')
        self.data_format = normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        self.padding='same'


    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], self.output_dim[0], self.output_dim[1])
        elif self.data_format == 'channels_last':
            return (input_shape[0], self.output_dim[0], self.output_dim[1], input_shape[3])

    def _resize_fun(self, inputs, data_format):
        try:
            assert keras.backend.backend() == 'tensorflow'
            assert self.data_format == 'channels_last'
        except AssertionError:
            print("Only tensorflow backend is supported for the resize layer and accordingly 'channels_last' ordering")
        output = tf.image.resize_images(inputs, self.output_dim)
        return output

    def call(self, inputs):
        output = self._resize_fun(inputs=inputs, data_format=self.data_format)
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
           #       'padding': self.padding,
                  'data_format': self.data_format}
        base_config = super(ResizeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))