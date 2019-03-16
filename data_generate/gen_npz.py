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

def WriteToNpz(face_size, face_is_gray):
    if face_is_gray:
        face_channel = 1
    else:
        face_channel = 3
    data = []
    for indir_i in INDIR_LIST:
        box_dir_i = os.path.join(indir_i, "box")
        landmark_dir_i = os.path.join(indir_i, "landmark")
        image_path_i = os.path.join(indir_i, "images")
        data_i = getDataFromJson(image_path_i, box_dir_i, landmark_dir_i, landmarks_num)
        data.extend(data_i)

    idx = 0
    total_data_num = len(data)
    dataset_array = np.zeros(shape=(total_data_num, face_size, face_size, face_channel))
    pts_array = np.zeros(shape=(total_data_num, landmarks_num, 2))
    for (imgPath, bbox, landmarkGt) in data:
        img_name = (imgPath.split("/")[-1]).split(".")[-2]
        filename = img_name.encode('utf8')

        img = cv2.imread(imgPath)
        assert (img is not None)
        if face_is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
        f_face = img[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]
        f_face = cv2.resize(f_face, (face_size, face_size))
        f_face = f_face.reshape(face_size, face_size, face_channel)
        assert len(f_face.shape) == 3
        height = f_face.shape[0]
        width = f_face.shape[1]
        if face_is_gray:
            assert f_face.shape[2] == 1
        else:
            assert f_face.shape[2] == 3

        dataset_array[idx, :, :, :] = f_face

        points_data = landmarkGt.reshape(landmarks_num*2)
        new_points_data = extract_points(points_data, landmarks_num, index_extract)
        # normalize
        norm_points_data = get_valid_points(gt_box, new_points_data.reshape([-1, 2]))
       # norm_points_data_list = norm_points_data.reshape(len(norm_points_data)*2).tolist()

        pts_array[idx, :, :] = norm_points_data

        if (idx % 100 == 0):
            sys.stdout.write('\r>> Data stacking %d/%d' % (idx + 1, total_data_num))
            sys.stdout.flush()

        idx += 1

    print("\nCreate Image array!")
    print("Create Points array!")

    # Save Image and Points array
    np.savez_compressed(OUTDIR + "img_dataset.npz", dataset_array)
    np.savez_compressed(OUTDIR + "pts_dataset.npz", pts_array)
    print("Save Image and Points array")


if __name__ == '__main__':
    INDIR_LIST = ["/home/jerry/disk/data/face_keypoint_detect/points_68/"]
    OUTDIR = "../samples/"
    crop_face_size = 128
    crop_face_is_gray = True
    landmarks_num = 68
   # index_extract = [36, 39, 42, 45, 30, 48, 54, 62, 66]
    index_extract = []

    WriteToNpz(crop_face_size, crop_face_is_gray)