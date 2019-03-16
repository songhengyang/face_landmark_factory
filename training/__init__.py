import tensorflow as tf
import numpy as np
import cv2
from keras.callbacks import Callback

class EvaluateInputTensor(Callback):
    """ Validate a model which does not expect external numpy data during training.

    Keras does not expect external numpy data at training time, and thus cannot
    accept numpy arrays for validation when all of a Keras Model's
    `Input(input_tensor)` layers are provided an  `input_tensor` parameter,
    and the call to `Model.compile(target_tensors)` defines all `target_tensors`.
    Instead, create a second model for validation which is also configured
    with input tensors and add it to the `EvaluateInputTensor` callback
    to perform validation.

    It is recommended that this callback be the first in the list of callbacks
    because it defines the validation variables required by many other callbacks,
    and Callbacks are made in order.

    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, model, steps, metrics_prefix='val', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        super(EvaluateInputTensor, self).__init__()
        self.val_model = model
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

    def on_epoch_end(self, epoch, logs={}):
        self.val_model.set_weights(self.model.get_weights())
        results = self.val_model.evaluate(None, None, steps=int(self.num_steps),
                                          verbose=self.verbose)
        metrics_str = '\n'
        for result, name in zip(results, self.val_model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)

def read_tfrecord(tfrecord_file, image_size, landmark_num, batch_size, is_train):
    filename_queue = tf.train.string_input_producer(tfrecord_file, shuffle=True)
    # read tfrecord
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'label/landmark': tf.FixedLenFeature([landmark_num*2], tf.float32)
        }
    )

    image_decoded = tf.decode_raw(features['image/encoded'], tf.uint8)
    if image_size < 224:
        image_shape = (image_size, image_size, 1)
    elif image_size >= 224:
        image_shape = (224, 224, 1)
        print("image_size > 224, it should be less than 224 !")
    image = tf.reshape(image_decoded, shape=image_shape)
    image = tf.cast(image, tf.float32)
   # image = (tf.cast(image, tf.float32) - 127.5) / 128
    # image = tf.image.per_image_standardization(image)
    landmark = tf.cast(features['label/landmark'], tf.float32)

    if is_train:
        image_batch, landmark_batch = tf.train.shuffle_batch(
            tensors=[image, landmark],
            batch_size=batch_size,
            num_threads=2,
            capacity=10000,
            min_after_dequeue=3000,
            enqueue_many=False
        )
    else:
        image_batch, landmark_batch = tf.train.batch(
            tensors=[image, landmark],
            batch_size=batch_size,
            num_threads=2,
            capacity=10000,
            enqueue_many=False)
   # image_batch = tf.reshape(image, shape=batch_shape)
   # landmark_batch = tf.reshape(landmark, [batch_size, landmark_num*2])

    return image_batch, landmark_batch