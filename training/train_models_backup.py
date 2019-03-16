from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import keras
from keras.models import Model
from keras.layers import Dense
from keras.utils import plot_model
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from keras.utils import multi_gpu_model
# Own module
from training import read_tfrecord, EvaluateInputTensor
from testing.utils import smoothL1, relu6
from net.all_facial_landmark_net import *

def main(facial_landmark_net):
#        Define X and y
# #        Load data
        if LOAD_TFRECORD:
            tfrecord_train_file = [os.path.join(DATA_DIR, "train.tfrecords"),
                                   os.path.join(DATA_DIR, "augment_train.tfrecords")]
            tfrecord_test_file = [os.path.join(DATA_DIR, "test.tfrecords"),
                                  os.path.join(DATA_DIR, "augment_test.tfrecords")]
            X_train_batch, y_train_batch = read_tfrecord(tfrecord_train_file, IMAGE_SIZE, N_LANDMARK, BATCH_SIZE, is_train=True)
            X_test_batch, y_test_batch = read_tfrecord(tfrecord_test_file, IMAGE_SIZE, N_LANDMARK, BATCH_SIZE, is_train=False)
        else:
            X = np.load(DATA_DIR + "img_dataset.npz")
            y = np.load(DATA_DIR + "pts_dataset.npz")
            X = X['arr_0']
            y = y['arr_0'].reshape(-1, OUTPUT_SIZE)

            print("Define X and Y")
            print("=======================================")

            # Split train / test dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            print("Success of getting train / test dataset")
            print("=======================================")
            print("X_train: ", X_train.shape)
            print("y_train: ", y_train.shape)
            print("X_test: ", X_test.shape)
            print("y_test: ", y_test.shape)
            print("=======================================")

        '''========define net================================'''
        if LOAD_TFRECORD:
            model = globals()[facial_landmark_net](input_shape=None, input_tensor=X_train_batch, output_size=OUTPUT_SIZE)
            test_model = globals()[facial_landmark_net](input_shape=None, input_tensor=X_test_batch, output_size=OUTPUT_SIZE)

        else:
            model = globals()[facial_landmark_net](input_shape=INPUT_SHAPE, input_tensor=None, output_size=OUTPUT_SIZE)
        save_model = globals()[facial_landmark_net](input_shape=INPUT_SHAPE, input_tensor=None, output_size=OUTPUT_SIZE)

        # Drawing model diagram
        modelname = facial_landmark_net
        model_picture_file = "./" + modelname + ".png"
        plot_model(model, to_file=model_picture_file)
        '''==================================================='''

        if LOAD_TFRECORD:
            model.compile(loss=smoothL1, optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['mape'], target_tensors=[y_train_batch])
            test_model.compile(loss=smoothL1, optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['mape'], target_tensors=[y_test_batch])
        else:
            try:
                parallel_model = multi_gpu_model(model, cpu_relocation=True)
                print("Training using multiple GPUs..")
            except ValueError:
                parallel_model = model
                print("Training using single GPU or CPU..")
            parallel_model.compile(loss=smoothL1, optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['mape'])
        save_model.compile(loss=smoothL1, optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['mape'])
        print(model.summary())

        # load pretrained weights
        checkpoint_file_dir = os.path.join("../log/", modelname)
        if not os.path.exists(checkpoint_file_dir):
            os.mkdir(checkpoint_file_dir)
        checkpoint_file = os.path.join(checkpoint_file_dir, CHECKPOINT_FILE_NAME)
        if os.path.exists(checkpoint_file):
            model.load_weights(checkpoint_file)

        # checkpoint
        filepath = os.path.join(checkpoint_file_dir, "smooth_L1-{epoch:02d}-{val_mean_absolute_percentage_error:.5f}.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        print("Start training...")
        #train model
        if LOAD_TFRECORD:
            # Fit the model using data from the TFRecord data tensors.
            coord = tf.train.Coordinator()
            sess = K.get_session()
            threads = tf.train.start_queue_runners(sess, coord)
            history = model.fit(epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                                callbacks=[EvaluateInputTensor(test_model, steps=TEST_STEPS),checkpoint])
        else:
            history = parallel_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
                                verbose=1, validation_data=(X_test, y_test), callbacks=callbacks_list)

        # Save model
        save_file = "../model/" + modelname + ".h5"
        model.save(save_file)
        save_model.load_weights(save_file)
        save_model.save(save_file)
        print("=======================================")
        print("Save Final Model")
        print("=======================================")

        if LOAD_TFRECORD:
            # Clean up the TF session.
            coord.request_stop()
            coord.join(threads)
            K.clear_session()

if __name__ == "__main__":
    LOAD_TFRECORD = True
   # DATA_DIR = "/home/jerry/disk/facelandmark_trainset/npz/64_64_1/"
    DATA_DIR = "/home/jerry/disk/facelandmark_trainset/tfrecord/64_64_1"
    net_list = ["facial_landmark_cnn", "facial_landmark_MobileNet", "facial_landmark_MobileNetV2",
                "facial_landmark_NASNetMobile", "facial_landmark_SqueezeNet", "facial_landmark_ResNet50",
                "facial_landmark_Xception", "facial_landmark_ResNeXt50"]
    NET = 2  # 0--basicnet  1--mobilenet_v1  2--mobilenet_v2  3--nasnet_mobile  4--squeezenet  5--resnet50
             # 6--xception  7--resnext50
    CHECKPOINT_FILE_NAME = "smooth_L1-294-0.86481.hdf5"
    BATCH_SIZE = 60
    STEPS_PER_EPOCH = 123713//BATCH_SIZE + 1
    TEST_STEPS = 13746//BATCH_SIZE + 1
    EPOCHS = 1
    IMAGE_SIZE = 64  #<=224
    INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
    N_LANDMARK = 68
    OUTPUT_SIZE = N_LANDMARK*2
    main(net_list[NET])
