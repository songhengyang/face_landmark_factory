# coding: utf-8
import os
import random
from os.path import join, exists
import tensorflow as tf
import cv2
import numpy as np
import numpy.random as npr
import sys
import io
from data_generate.BBox_utils import getDataFromJson

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def extract_points(input_data, input_points_num, index_extract):
    output_data = []
    if len(input_data) == input_points_num*2 and len(index_extract) != 0:
        # add pupile points
        for i in range(0, len(index_extract)):
            point_temp_x = input_data[index_extract[i]*2]
            point_temp_y = input_data[index_extract[i]*2+1]
            output_data.append(float(point_temp_x))
            output_data.append(float(point_temp_y))
        assert len(output_data) == len(index_extract)*2, "The new landmarks should contain %s points." % len(index_extract)
    else:
        output_data = input_data
    return np.array(output_data)

def get_valid_points(box, points):
    """Update points locations according to new image size"""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    width = right_x - left_x
    height = bottom_y - top_y

    # Shift points first.
    for point in points:
        point[0] -= left_x
        point[1] -= top_y

    # Then normalize the coordinates.
    for point in points:
        point[0] /= width
        point[1] /= height

    return points

def WriteToTfrecord(indir_list, face_size, landmarks_num, is_gray, ratio, tfrecord_file_dir, index_extract):
    index_num = len(index_extract)
    if index_num == 0:
        index_num = landmarks_num
    train_tfrecord_file = os.path.join(tfrecord_file_dir, "train_%sp.tfrecords" % index_num)
    test_tfrecord_file = os.path.join(tfrecord_file_dir, "test_%sp.tfrecords" % index_num)
    data = []
    for indir_i in indir_list:
        box_dir_i = os.path.join(indir_i, "box")
        landmark_dir_i = os.path.join(indir_i, "landmark")
        image_path_i = os.path.join(indir_i, "images")
        data_i = getDataFromJson(image_path_i, box_dir_i, landmark_dir_i, landmarks_num)
        data.extend(data_i)
    tf_train_writer = tf.python_io.TFRecordWriter(train_tfrecord_file)
    tf_test_writer = tf.python_io.TFRecordWriter(test_tfrecord_file)
    idx = 0
    total_data_num = len(data)
    test_data_num = round(total_data_num*ratio)
    train_data_num = total_data_num - test_data_num
    f_txt_file = os.path.join(tfrecord_file_dir, "count_%sp.txt" % index_num)
    f_txt = open(f_txt_file, 'w')
    f_txt.write("total data number:%d\n" % total_data_num)
    f_txt.write("train data number:%d\n" % train_data_num)
    f_txt.write("test data number:%d\n" % test_data_num)
    f_txt.close()
    for (imgPath, bbox, landmarkGt) in data:
        img_name = (imgPath.split("/")[-1]).split(".")[-2]
        filename = img_name.encode('utf8')

        # Read the image.
        head, tail = os.path.split(imgPath)
        image_name = tail.split('.')[-2]
        img_jpg = os.path.join(head, image_name + ".jpg")
        img_png = os.path.join(head, image_name + ".png")
        if os.path.exists(img_jpg):
            img = cv2.imread(img_jpg)
        else:
            img = cv2.imread(img_png)
        assert img is not None, print("image can't is opened! image dir = %s", imgPath)

        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
        f_face = img[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]
        f_face = cv2.resize(f_face, (face_size, face_size))
        if is_gray:
            f_face = cv2.cvtColor(f_face, cv2.COLOR_BGR2GRAY)
        '''
        f_face_data = np.zeros(face_size*face_size, dtype=np.uint8)
        for i in range(face_size):
            for j in range(face_size):
                f_face_data[i*face_size + j] = f_face.item(i, j)
        '''
        # transform data into string format
        image_data = f_face.tostring()

        points_data = landmarkGt.reshape(landmarks_num*2)
        new_points_data = extract_points(points_data, landmarks_num, index_extract)
        # normalize
        norm_points_data = get_valid_points(gt_box, new_points_data.reshape([-1, 2]))
        norm_points_data_list = norm_points_data.reshape(len(norm_points_data)*2).tolist()

        if (idx % 100 == 0):
            sys.stdout.write('\r>> Converting image %d/%d' % (idx + 1, total_data_num))
            sys.stdout.flush()

        current_example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_data),
        'label/landmark': _float32_feature(norm_points_data_list),
        }))
        if idx < test_data_num:
            tf_test_writer.write(current_example.SerializeToString())
        else:
            tf_train_writer.write(current_example.SerializeToString())
        idx += 1

    tf_train_writer.close()
    tf_test_writer.close()


if __name__ == '__main__':

    input_list = ["../data/json"]

    '''
    input_list = ["/home/jerry/disk/data/face_keypoint_detect/points_70/ibug",
                  "/home/jerry/disk/data/face_keypoint_detect/points_70/lfpw/train",
                  "/home/jerry/disk/data/face_keypoint_detect/points_70/lfpw/test",
                  "/home/jerry/disk/data/face_keypoint_detect/points_70/helen/train",
                  "/home/jerry/disk/data/face_keypoint_detect/points_70/helen/test",
                  "/home/jerry/disk/data/face_keypoint_detect/points_70/afw",
                  "/home/jerry/disk/data/face_keypoint_detect/points_70/300W/Indoor",
                  "/home/jerry/disk/data/face_keypoint_detect/points_70/300W/Outdoor"]

     base_dir_list = ["/home/jerry/disk/data/face_keypoint_detect/points_70/300VW",
                     "/home/jerry/disk/data/face_keypoint_detect/points_70/longtianVW"]

    for base_dir_i in base_dir_list:
        sub_paths = os.listdir(base_dir_i)
        for sub_path in sub_paths:
            input_list.append(os.path.join(base_dir_i, sub_path))
    '''

    tfrecord_file_dir = "../data/tfrecord"

    crop_face_size = 64
    landmarks_num = 68
    is_gray = True
    ratio = 0.1
#    index_extract = [36, 39, 42, 45, 30, 48, 54, 62, 66]
    index_extract = []
    if not os.path.exists(tfrecord_file_dir):
        os.makedirs(tfrecord_file_dir)

    WriteToTfrecord(input_list,
                    crop_face_size,
                    landmarks_num,
                    is_gray,
                    ratio,
                    tfrecord_file_dir,
                    index_extract)
