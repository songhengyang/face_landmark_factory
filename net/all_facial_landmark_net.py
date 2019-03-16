import tensorflow as tf
from tensorflow import keras

from net.basicnet import basic_cnn
from net.mobilenet_v1 import MobileNet
from net.mobilenet_v2 import MobileNetV2
from net.nasnet import NASNetMobile
from net.squeezenet import SqueezeNet
from net.resnet50 import ResNet50
from net.resnet_common import ResNeXt50
from net.xception import Xception

Dense = keras.layers.Dense
Model = keras.models.Model
plot_model = keras.utils.plot_model

def facial_landmark_cnn(input_shape, input_tensor, output_size):
    model = basic_cnn(input_shape=input_shape, input_tensor=input_tensor, output_size=output_size)
    return model

def facial_landmark_MobileNet(input_shape, input_tensor, output_size):
    model_pre = MobileNet(input_shape=input_shape, input_tensor=input_tensor, alpha=1.0, depth_multiplier=1,
                          include_top=False, weights=None, pooling='max', shallow=True)
    num_outputs = output_size
    last_layer = model_pre.get_layer('max_pool').output
    out = Dense(num_outputs, name='output')(last_layer)
    model = Model(model_pre.input, out)
    return model

def facial_landmark_MobileNetV2(input_shape, input_tensor, output_size):
    model_pre = MobileNetV2(input_shape=input_shape, input_tensor=input_tensor, alpha=1.0, depth_multiplier=1,
                            include_top=False, weights=None, pooling='max', shallow=True)
    num_outputs = output_size
    last_layer = model_pre.get_layer('max_pool').output
    out = Dense(num_outputs, name='output')(last_layer)
    model = Model(model_pre.input, out)
    return model

def facial_landmark_NASNetMobile(input_shape, input_tensor, output_size):
    model_pre = NASNetMobile(input_shape=input_shape, input_tensor=input_tensor, include_top=False,
                             weights=None, pooling='max', shallow=True)
    num_outputs = output_size
    last_layer = model_pre.get_layer('max_pool').output
    out = Dense(num_outputs, name='output')(last_layer)
    model = Model(model_pre.input, out)
    return model

def facial_landmark_SqueezeNet(input_shape, input_tensor, output_size):
    model_pre = SqueezeNet(input_shape=input_shape, input_tensor=input_tensor, include_top=False,
                           pooling='max', shallow=True)
    num_outputs = output_size
    last_layer = model_pre.get_layer('max_pool').output
    out = Dense(num_outputs, name='output')(last_layer)
    model = Model(model_pre.input, out)
    return model

def facial_landmark_ResNet50(input_shape, input_tensor, output_size):
    model_pre = ResNet50(input_shape=input_shape, input_tensor=input_tensor, include_top=False,
                         weights=None, pooling='max', shallow=True)
    num_outputs = output_size
    last_layer = model_pre.get_layer('max_pool').output
    out = Dense(num_outputs, name='output')(last_layer)
    model = Model(model_pre.input, out)
    return model

def facial_landmark_Xception(input_shape, input_tensor, output_size):
    model_pre = Xception(input_shape=input_shape, input_tensor=input_tensor, include_top=False,
                         weights=None, pooling='max', shallow=True)
    num_outputs = output_size
    last_layer = model_pre.get_layer('max_pool').output
    out = Dense(num_outputs, name='output')(last_layer)
    model = Model(model_pre.input, out)
    return model

def facial_landmark_ResNeXt50(input_shape, input_tensor, output_size):
    model_pre = ResNeXt50(input_shape=input_shape, input_tensor=input_tensor, include_top=False,
                          weights=None, pooling='max', shallow=True)
    num_outputs = output_size
    last_layer = model_pre.get_layer('max_pool').output
    out = Dense(num_outputs, name='output')(last_layer)
    model = Model(model_pre.input, out)
    return model

if __name__ == '__main__':
    model = facial_landmark_SqueezeNet(input_shape=(60,60,1), input_tensor=None, output_size = 136)
    plot_model(model, to_file='./facial_landmark_SqueezeNet.png')
