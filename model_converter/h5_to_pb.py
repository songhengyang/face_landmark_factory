import tensorflow as tf
from tensorflow.python.framework import graph_io
from testing.utils import smoothL1
from net.all_facial_landmark_net import *


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

"""----------------------------------配置路径-----------------------------------"""
net_list = ["facial_landmark_cnn", "facial_landmark_MobileNet", "facial_landmark_MobileNetV2",
                "facial_landmark_NASNetMobile", "facial_landmark_SqueezeNet", "facial_landmark_ResNet50",
                "facial_landmark_Xception", "facial_landmark_ResNeXt50"]
NET = 4  # 0--basicnet  1--mobilenet_v1  2--mobilenet_v2  3--nasnet_mobile  4--squeezenet  5--resnet50
             # 6--xception  7--resnext50
facial_landmark_net = net_list[NET]
input_file = "../model/%s.h5" % facial_landmark_net
output_path = "../model/"
pb_model_name = "%s.pb" % facial_landmark_net
method = 0
"""----------------------------------导入keras模型------------------------------"""
if method == 0:
    with tf.keras.utils.CustomObjectScope({'relu6': tf.keras.layers.ReLU(6.),
                            'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D,
                            'smoothL1': smoothL1}):
        tf.keras.backend.set_learning_phase(0)
        net_model = tf.keras.models.load_model(input_file)
else:
    image_size = 64
    n_landmark = 68
    input_shape = (image_size, image_size, 1)
    output_size = n_landmark * 2
    net_model = globals()[facial_landmark_net](input_shape=input_shape, input_tensor=None, output_size=output_size)
    net_model.load_weights(input_file)

print('input is :', net_model.input.name)
print('output is:', net_model.output.name)

"""----------------------------------保存为.pb格式------------------------------"""
sess = tf.keras.backend.get_session()
frozen_graph = freeze_session(sess, output_names=[net_model.output.op.name])
graph_io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)