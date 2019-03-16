from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import tensorflow as tf
# Own module
from testing.utils import smoothL1, relu6
from net.all_facial_landmark_net import *
import multiprocessing as mt
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


def plot_history(history, result_dir):
    '''
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()
    '''
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
   # acc = history.history['acc']
    val_loss = history.history['val_loss']
   # val_acc = history.history['val_acc']
    nb_epoch = len(loss)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tval_loss\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\n'.format(
                i, loss[i], val_loss[i]))
        fp.close()

def parse_exmp(serial_exmp):
    features = tf.parse_single_example(
        serial_exmp,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'label/landmark': tf.FixedLenFeature([N_LANDMARK * 2], tf.float32)
        }
    )
    image_decoded = tf.decode_raw(features['image/encoded'], tf.uint8)
    if IMAGE_SIZE < 224:
        image_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)
    elif IMAGE_SIZE >= 224:
        image_shape = (224, 224, 1)
        print("image_size > 224, it should be less than 224 !")
    image = tf.reshape(image_decoded, shape=image_shape)
    image = tf.cast(image, tf.float32)
    # image = (tf.cast(image, tf.float32) - 127.5) / 128
    # image = tf.image.per_image_standardization(image)
    landmark = tf.cast(features['label/landmark'], tf.float32)
    return image, landmark

def get_dataset(record_name, batch_size, epochs):
    dataset = tf.data.TFRecordDataset(record_name)
    dataset = dataset.map(parse_exmp, mt.cpu_count())
    dataset = dataset.repeat(epochs).shuffle(1000).batch(batch_size)
    return dataset


def main(facial_landmark_net):
# #        Load data
        weights_name = "weights"
        tfrecord_train_file = [os.path.join(DATA_DIR, "train_%sp.tfrecords" % N_LANDMARK)]#,
                               #os.path.join(DATA_DIR, "augment_train.tfrecords")]
        tfrecord_test_file = [os.path.join(DATA_DIR, "test_%sp.tfrecords" % N_LANDMARK)]#,
                              #os.path.join(DATA_DIR, "augment_test.tfrecords")]
        train_batch = get_dataset(tfrecord_train_file, BATCH_SIZE, EPOCHS)
        test_batch = get_dataset(tfrecord_test_file, BATCH_SIZE, EPOCHS)
        '''========define net================================'''
        model = globals()[facial_landmark_net](input_shape=INPUT_SHAPE, input_tensor=None, output_size=OUTPUT_SIZE)
        save_model = globals()[facial_landmark_net](input_shape=INPUT_SHAPE, input_tensor=None, output_size=OUTPUT_SIZE)

        # Drawing model diagram
        model_name = facial_landmark_net
        model_picture_file = "./" + model_name + ".png"
        tf.keras.utils.plot_model(model, to_file=model_picture_file)
        '''==================================================='''
        try:
            parallel_model = tf.keras.utils.multi_gpu_model(model, gpus=2)
            print("Training using multiple GPUs..")
        except ValueError:
            parallel_model = model
            print("Training using single GPU or CPU..")
        parallel_model.compile(loss=smoothL1, optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['mape'])
        save_model.compile(loss=smoothL1, optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['mape'])
        print(model.summary())

        # load pretrained weights
        checkpoint_file_dir = os.path.join("../log/", model_name)
        if not os.path.exists(checkpoint_file_dir):
            os.mkdir(checkpoint_file_dir)
        checkpoint_file = os.path.join(checkpoint_file_dir, weights_name)
        if LOADING_PRETRAIN:
            model.load_weights(checkpoint_file)

        # checkpoint
        filepath = os.path.join(checkpoint_file_dir, weights_name) #weights.{epoch:02d}-{val_loss:.2f}.hdf5
        callbacks = [
            # Interrupt training if `val_loss` stops improving for over 2 epochs
           # tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir=checkpoint_file_dir),
            tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                               monitor='val_loss',
                                               verbose=1,
                                               save_best_only=True,
                                               save_weights_only=False,
                                               mode='auto',
                                               period=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.1,
                                                 patience=3,
                                                 verbose=1)
        ]
        print("Start training...")
        #train model
        history = parallel_model.fit(train_batch, batch_size=None, epochs=EPOCHS,
                                     steps_per_epoch=STEPS_PER_EPOCH, shuffle=True,
                                     verbose=1, validation_data=test_batch,
                                     validation_steps=TEST_STEPS, callbacks=callbacks)
        plot_history(history, checkpoint_file_dir)
        save_history(history, checkpoint_file_dir)
        # Save model
        save_file_keras = "../model/" + model_name + ".h5"
        save_file_tf = os.path.join(checkpoint_file_dir, weights_name)
       # model.save(save_file_keras)
        model.save_weights(save_file_tf, save_format='tf')
        save_model.load_weights(save_file_tf)
        save_model.save(save_file_keras)
        print("=======================================")
        print("Save Final Model")
        print("=======================================")


if __name__ == "__main__":
    DATA_DIR = "../data/tfrecord"
    net_list = ["facial_landmark_cnn", "facial_landmark_MobileNet", "facial_landmark_MobileNetV2",
                "facial_landmark_NASNetMobile", "facial_landmark_SqueezeNet", "facial_landmark_ResNet50",
                "facial_landmark_Xception", "facial_landmark_ResNeXt50"]
    NET = 4  # 0--basicnet  1--mobilenet_v1  2--mobilenet_v2  3--nasnet_mobile  4--squeezenet  5--resnet50
             # 6--xception  7--resnext50
    LOADING_PRETRAIN = False
    BATCH_SIZE = 10
    STEPS_PER_EPOCH = 100//BATCH_SIZE
    TEST_STEPS = 11//BATCH_SIZE
    EPOCHS = 1000
    IMAGE_SIZE = 64  #<=224
    INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
    N_LANDMARK = 68
    OUTPUT_SIZE = N_LANDMARK*2

    main(net_list[NET])
