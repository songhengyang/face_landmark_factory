from keras import backend as K
from keras.models import Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Reshape, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
from keras import initializers, regularizers, constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras.applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions, _obtain_input_shape
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import keras

import numpy as np

# Our Own Loss Function
HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  K.sum(x)

weights = np.empty((136,)) # Outer: Brows: Nose: Eyes: Mouth (0.5 : 1 : 2 : 3 : 1)

weights[0:33] = 3
weights[33:53] = 1
weights[53:71] = 2
weights[71:95] = 2
weights[95:] = 1

def mask_weights(y_true, y_pred):
    x = K.abs(y_true - y_pred) * weights
    return K.sum(x)

def relu6(x):
    return K.relu(x, max_value=6)

class DepthwiseConv2D(Conv2D):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0], self.kernel_size[1],
                                  input_dim, self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(input_dim * self.depth_multiplier, ),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding, self.strides[1])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(
            self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(
            self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(
            self.depthwise_constraint)
        return config
        
# Tracker
import cv2
import os
import numpy as np

class BboxTracker():
    def __init__(self, args):
        self.args = args
        # self.tracker = cv2.TrackerMedianFlow_create()
        self.tracker = cv2.TrackerKCF_create()
    
    def xyxy2xywh(self, bbox_xyxy):
        x = bbox_xyxy[0][0]
        y = bbox_xyxy[0][1]
        w = bbox_xyxy[1][0] - bbox_xyxy[0][0]
        h = bbox_xyxy[1][1] - bbox_xyxy[0][1]
        return (x,y,w,h)

    def xywh2xyxy(self, bbox_xywh):
        x1 = int(bbox_xywh[0])
        y1 = int(bbox_xywh[1])
        x2 = int(bbox_xywh[0] + bbox_xywh[2])
        y2 = int(bbox_xywh[1] + bbox_xywh[3])
        return [(x1,y1),(x2,y2)]

    def initTracker(self, img, bbox):
        bbox_xywh = self.xyxy2xywh(bbox)
        self.tracker = cv2.TrackerKCF_create()
        # self.tracker = cv2.TrackerMedianFlow_create()
        ok = self.tracker.init(img, bbox_xywh)
        return ok
    
    def updateBbox(self, img):
        ok, bbox_xywh = self.tracker.update(img)
        if ok:
            return self.xywh2xyxy(bbox_xywh)
        else:
            return None

# import the necessary packages
import datetime
 
class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
 
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
 
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
 
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
 
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
 
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()