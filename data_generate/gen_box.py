"""
This script shows how to read iBUG pts file and draw all the landmark points on image.
"""
import json
import os
import numpy as np
import cv2
import sys

from data_generate.BBox_utils import read_points_from_json, read_points_from_pts
from data_generate.Landmark_utils import points_are_valid, get_valid_box, get_minimal_box, draw_landmark_point
from data_generate.face_detector import draw_box

def gen_box(point_file, verbose=False):
    # Read the points from file.
    landmarks = read_points_from_json(point_file)
    # Safe guard, make sure point importing goes well.
    assert len(landmarks) == LANDMARKS_NUM, "The landmarks should contain %d points." % LANDMARKS_NUM

    # Read the image.
    head, tail = os.path.split(point_file)
    image_name = tail.split('.')[-2]
    image_dir = os.path.join(os.path.split(head)[0], "images")
    img_jpg = os.path.join(image_dir, image_name + ".jpg")
    img_png = os.path.join(image_dir, image_name + ".png")
    if os.path.exists(img_jpg):
        img = cv2.imread(img_jpg)
    elif os.path.exists(img_png):
        img = cv2.imread(img_png)
    else:
        print("image can't is opened! image dir = %s, image name = %s" % (image_dir, image_name))
        return None

    # Fast check: all points are in image.
    if points_are_valid(landmarks, img) is False:
        print("Invalid pts file, ignored:", point_file)
        return None

    # Get the valid facebox.
    facebox = get_valid_box(img, landmarks)
    box_color = (255, 0, 0)
    if facebox is None:
        print("Using minimal box.")
        facebox = get_minimal_box(landmarks)
        box_color = (0, 0, 255)

    # Resize image and compute local points.
    width = facebox[2] - facebox[0]
    height = facebox[3] - facebox[1]
    if width != height:
        print('opps!', width, height)

    # Save the box location.
    box_dir = os.path.join(os.path.split(head)[0], "box")
    box_file = os.path.join(box_dir, image_name + ".json")
    with open(box_file, mode='w') as file2:
        json.dump(facebox, file2)

   # print("New file saved:", landmark_json_url, box_json_url, image_json_url)
    if verbose:
        # Show the result.
        draw_box(img, [facebox], box_color)
        draw_landmark_point(img, landmarks)
        # Show whole image in window.
        width, height = img.shape[:2]
        max_height = 640
        if height > max_height:
            img = cv2.resize(
                img, (max_height, int(width * max_height / height)))
        window_name = "annotation image"
        cv2.imshow(window_name, img)
        cv2.waitKey(10)
       # cv2.destroyWindow(window_name)

def main(input_list, verbose=False):
    # List all the pts files
    json_file_list = []
    for indir_i in input_list:
        box_dir = os.path.join(indir_i, "box")
        if not os.path.exists(box_dir):
            os.mkdir(box_dir)
        assert os.path.exists(box_dir)
        landmark_dir = os.path.join(indir_i, "annotations")
        for file_path, _, file_names in os.walk(landmark_dir):
            for file_name in file_names:
                if file_name.split(".")[-1] in ["json"]:
                    json_file_list.append(os.path.join(file_path, file_name))

    # Show the image one by one.
    total_data_num = len(json_file_list)
    data_index = 0
    for json_file_i in json_file_list:
       # print("The %sth graph are currently processed." % images_index)
        if (data_index % 10 == 0):
            sys.stdout.write('\r>> Processing image %d/%d' % (data_index + 1, total_data_num))
            sys.stdout.flush()

        gen_box(point_file=json_file_i, verbose=verbose)
        data_index += 1

if __name__ == "__main__":
    input_list = ["../data/json"]
    base_dir = "/home/jerry/disk/data/face_keypoint_detect/points_70/300VW"
    sub_paths = os.listdir(base_dir)
    for sub_path in sub_paths:
        input_list.append(os.path.join(base_dir, sub_path))
        
    LANDMARKS_NUM = 70
    main(input_list=input_list, verbose=False)
    '''
    input_list = "/home/jerry/disk/data/face_keypoint_detect/points_70/300VW"
    sub_1s = os.listdir(input_list)
    for sub_1_i in sub_1s:
        sub_1_i_dir = os.path.join(input_list, sub_1_i)
        sub_2s = os.listdir(sub_1_i_dir)
        for sub_2_i in sub_2s:
            sub_2_i_dir = os.path.join(sub_1_i_dir, sub_2_i)
            if sub_2_i == "image":
                new_dir = os.path.join(sub_1_i_dir, "images")
                os.rename(sub_2_i_dir, new_dir)
    '''