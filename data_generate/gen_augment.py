# coding: utf-8
import os
import random
import json
import cv2
import numpy as np
import numpy.random as npr
import sys
import tensorflow as tf
from data_generate.BBox_utils import getDataFromJson, BBox, drawLandmarkBox, drawLandmark
from data_generate.Landmark_utils import rotate, flip, show_landmark
from data_generate.from_pts_to_json_box_image import points_are_valid, get_valid_box, get_minimal_box
from data_generate.gen_tfrecord import _int64_feature, _float32_feature, _bytes_feature


def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter * 1.0 / (box_area + area - inter)
    return ovr

def writeData(file_name, image, points, target_landmarks_dir, target_image_dir):
    new_points = points

    # New file to be written.
    common_file_name = file_name
    landmark_common_url = os.path.join(
        target_landmarks_dir, common_file_name)
    '''
    box_common_url = os.path.join(
        target_box_dir, common_file_name)
    '''
    image_common_url = os.path.join(
        target_image_dir, common_file_name)

    # Save the landmarks location.
    landmark_json_url = landmark_common_url + ".json"
    points_to_save = np.array(new_points).flatten()
    with open(landmark_json_url, mode='w') as file1:
        json.dump(list(points_to_save), file1)
    '''
    # Save the box location.
    box_json_url = box_common_url + ".json"
    with open(box_json_url, mode='w') as file2:
        json.dump(facebox, file2)
    '''
    # Save the image
    image_json_url = image_common_url + ".jpg"
    cv2.imwrite(image_json_url, image)


def GenerateData(size, landmarks_num, is_gray=True, verbose=False, is_tfrecord=True):
    data = []
    for indir_i in INDIR_LIST:
        box_dir_i = os.path.join(indir_i, "box")
        landmark_dir_i = os.path.join(indir_i, "landmark")
        image_path_i = os.path.join(indir_i, "images")
        data_i = getDataFromJson(image_path_i, box_dir_i, landmark_dir_i, landmarks_num)
        data.extend(data_i)
    total_data_num = len(data)
    test_data_num = round(total_data_num * ratio)
    if is_tfrecord:
        tf_train_writer = tf.python_io.TFRecordWriter(TF_TRAIN_FILE)
        tf_test_writer = tf.python_io.TFRecordWriter(TF_TEST_FILE)
    augment_data_total = 0
    augment_train_num = 0
    augment_test_num = 0
    idx = 0
    for (imgPath, bbox, landmarkGt) in data:
        # print imgPath
        F_imgs = []
        F_landmarks = []
        img = cv2.imread(imgPath)
        image_cp = img.copy()
        assert (img is not None)
        img_h, img_w, img_c = img.shape
        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
      #  f_face = img[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]
      #  f_face = cv2.resize(f_face, (size, size))
        if verbose:
            drawLandmarkBox(image_cp, bbox, landmarkGt)
            width, height = image_cp.shape[:2]
            max_height = 480
            if height > max_height:
                image_cp = cv2.resize(image_cp, (max_height, int(width * max_height / height)))
            cv2.imshow("raw image", image_cp)
            cv2.waitKey(0)
        landmark = np.zeros((landmarks_num, 2))

        if idx % 100 == 0:
            sys.stdout.write('\r>> Processing raw image %d/%d' % (idx, total_data_num))
            sys.stdout.flush()
        idx = idx + 1

        x1, y1, x2, y2 = gt_box
        # gt's width
        gt_w = x2 - x1 + 1
        # gt's height
        gt_h = y2 - y1 + 1
        if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
            continue
        # random shift
        for i in range(10):
            bbox_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
            delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
            delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
            nx1 = int(max(x1 + gt_w / 2 - bbox_size / 2 + delta_x, 0))
            ny1 = int(max(y1 + gt_h / 2 - bbox_size / 2 + delta_y, 0))

            nx2 = nx1 + bbox_size
            ny2 = ny1 + bbox_size
            if nx2 > img_w or ny2 > img_h:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
            resized_im = cv2.resize(cropped_im, (size, size))
            # cal iou
            iou = IoU(crop_box, np.expand_dims(gt_box, 0))
            if iou > 0.65:
                F_imgs.append(resized_im)
                # normalize
                for index, one in enumerate(landmarkGt):
                    rv = ((one[0] - nx1) / bbox_size, (one[1] - ny1) / bbox_size)
                    landmark[index] = rv
                F_landmarks.append(landmark.reshape(landmarks_num*2))
                landmark = np.zeros((landmarks_num, 2))
                landmark_ = F_landmarks[-1].reshape(-1, 2)
                bbox = BBox([nx1, ny1, nx2, ny2])

                # mirror
                if random.choice([0, 1]) > 0:
                    face_flipped, landmark_flipped = flip(resized_im, landmark_)
                    face_flipped = cv2.resize(face_flipped, (size, size))
                    # c*h*w
                    F_imgs.append(face_flipped)
                    F_landmarks.append(landmark_flipped.reshape(landmarks_num*2))
                # rotate
                if random.choice([0, 1]) > 0:
                    face_rotated_by_alpha, landmark_rotated = rotate(img, bbox,
                                                                     bbox.reprojectLandmark(landmark_), 5)  # 逆时针旋转
                    # landmark_offset
                    landmark_rotated = bbox.projectLandmark(landmark_rotated)
                    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                    F_imgs.append(face_rotated_by_alpha)
                    F_landmarks.append(landmark_rotated.reshape(landmarks_num*2))

                    # flip
                    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                    face_flipped = cv2.resize(face_flipped, (size, size))
                    F_imgs.append(face_flipped)
                    F_landmarks.append(landmark_flipped.reshape(landmarks_num*2))

                    # inverse clockwise rotation
                if random.choice([0, 1]) > 0:
                    face_rotated_by_alpha, landmark_rotated = rotate(img, bbox,
                                                                     bbox.reprojectLandmark(landmark_), -5)  # 顺时针旋转
                    landmark_rotated = bbox.projectLandmark(landmark_rotated)
                    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                    F_imgs.append(face_rotated_by_alpha)
                    F_landmarks.append(landmark_rotated.reshape(landmarks_num*2))

                    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                    face_flipped = cv2.resize(face_flipped, (size, size))
                    F_imgs.append(face_flipped)
                    F_landmarks.append(landmark_flipped.reshape(landmarks_num*2))

        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
        # print F_imgs.shape
        # print F_landmarks.shape
        for i in range(len(F_imgs)):
            # print(image_id)
            if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                continue
            if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                continue
            if verbose:
                image_aug = F_imgs[i].copy()
                landmark_aug = F_landmarks[i].reshape(68,2)
                for index, one in enumerate(landmark_aug):
                    rv = (one[0]*size, one[1]*size)
                    landmark_aug[index] = rv
                drawLandmark(image_aug, landmark_aug)
                image_aug = cv2.resize(image_aug, (256, 256))
                cv2.imshow("aug image", image_aug)
                cv2.waitKey(0)
                cv2.destroyWindow("aug image")
            if is_gray:
                f_face = cv2.cvtColor(F_imgs[i], cv2.COLOR_BGR2GRAY)
            else:
                f_face = F_imgs[i]
            if is_tfrecord:
                image_data = f_face.tostring()
                norm_points_data_list = F_landmarks[i].tolist()
                current_example = tf.train.Example(features=tf.train.Features(feature={
                    'image/encoded': _bytes_feature(image_data),
                    'label/landmark': _float32_feature(norm_points_data_list),
                }))
                if idx < test_data_num:
                    tf_train_writer.write(current_example.SerializeToString())
                    augment_test_num = augment_test_num + 1
                else:
                    tf_test_writer.write(current_example.SerializeToString())
                    augment_train_num = augment_train_num + 1
            else:
                writeData(str(augment_data_total), f_face, F_landmarks[i],
                          TARGET_LANDMARKS_DIR, TARGET_IMAGE_DIR)
            augment_data_total = augment_data_total + 1

    f_txt_file = os.path.join(OUTPUT_DIR, "augment_count.txt")
    f_txt = open(f_txt_file, 'w')
    if is_tfrecord:
        tf_train_writer.close()
        tf_test_writer.close()
        f_txt.write("total augment data number:%d\n" % augment_data_total)
        f_txt.write("augment train data number:%d\n" % augment_train_num)
        f_txt.write("augment test data number:%d\n" % augment_test_num)
        print("\n%d augment images are generated, %d train data, %d test data, and converted to tfrecord file !"
              % (augment_data_total, augment_train_num, augment_test_num))
    else:
        f_txt.write("total augment data number:%d\n" % augment_data_total)
        print("\n%d augment images are generated !" % augment_data_total)
    if is_gray:
        channel = 1
    else:
        channel = 3
    print("image shape is: %d %d %d" % (size, size, channel))
    f_txt.write("augment image shape is: %d %d %d" % (size, size, channel))
    f_txt.close()
    if verbose:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    INDIR_LIST = ["../data/json"]
    OUTPUT_DIR = "../data/augment"
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    assert os.path.exists(OUTPUT_DIR)
    TARGET_LANDMARKS_DIR = os.path.join(OUTPUT_DIR, "landmark")
    TARGET_IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
    TF_TRAIN_FILE = os.path.join(OUTPUT_DIR, "augment_train.tfrecords")
    TF_TEST_FILE = os.path.join(OUTPUT_DIR, "augment_test.tfrecords")
    if not os.path.exists(TARGET_LANDMARKS_DIR):
        os.mkdir(TARGET_LANDMARKS_DIR)
    if not os.path.exists(TARGET_IMAGE_DIR):
        os.mkdir(TARGET_IMAGE_DIR)
    image_size = 64
    landmarks_num = 68
    ratio = 0.1
    GenerateData(image_size, landmarks_num, is_gray=True, verbose=False, is_tfrecord=True)

