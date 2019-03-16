"""
This script shows how to read iBUG pts file and draw all the landmark points on image.
"""
import json
import os
import numpy as np
import cv2
import sys

from data_generate.BBox_utils import read_points_from_json, read_points_from_pts
from data_generate.Landmark_utils import points_are_valid, get_valid_box, get_minimal_box

def write_pts_to_json(point_file, raw_points, target_landmarks_dir, target_box_dir, target_image_dir):
    """
    Preview points on image.
    """
    new_points = raw_points

    # Read the image.
    head, tail = os.path.split(point_file)
    image_file = tail.split('.')[-2]
    img_jpg = os.path.join(head, image_file + ".jpg")
    img_png = os.path.join(head, image_file + ".png")
    if os.path.exists(img_jpg):
        img = cv2.imread(img_jpg)
    else:
        img = cv2.imread(img_png)

    # Fast check: all points are in image.
    if points_are_valid(new_points, img) is False:
        print("Invalid pts file, ignored:", point_file)
        return None

    # Get the valid facebox.
    facebox = get_valid_box(img, new_points)
    box_color = (255, 0, 0)
    if facebox is None:
        print("Using minimal box.")
        facebox = get_minimal_box(new_points)
        box_color = (0, 0, 255)

    # Resize image and compute local points.
    width = facebox[2] - facebox[0]
    height = facebox[3] - facebox[1]
    if width != height:
        print('opps!', width, height)

    # New file to be written.
    head, tail = os.path.split(point_file)
    subset_name = head.split('/')[-2]
    common_file_name = tail.split('.')[-2]
    landmark_common_url = os.path.join(
        target_landmarks_dir, common_file_name)
    box_common_url = os.path.join(
        target_box_dir, common_file_name)
    image_common_url = os.path.join(
        target_image_dir, common_file_name)

    # Save the landmarks location.
    landmark_json_url = landmark_common_url + ".json"
    points_to_save = np.array(new_points).flatten()
    with open(landmark_json_url, mode='w') as file1:
        json.dump(list(points_to_save), file1)

    # Save the box location.
    box_json_url = box_common_url + ".json"
    with open(box_json_url, mode='w') as file2:
        json.dump(facebox, file2)

    # Save the image
    image_json_url = image_common_url + ".jpg"
    cv2.imwrite(image_json_url, img)

   # print("New file saved:", landmark_json_url, box_json_url, image_json_url)
'''
    # Show the result.
    fd.draw_box(img, [facebox], box_color)
    draw_landmark_point(img, new_points)
    # Show whole image in window.
    width, height = img.shape[:2]
    max_height = 640
    if height > max_height:
        img = cv2.resize(
            img, (max_height, int(width * max_height / height)))
    cv2.imshow("raw image", img)
'''


def main():
    # List all the pts files
    pts_file_list = []
    for indir_i in INPUT_LIST:
        for file_path, _, file_names in os.walk(indir_i):
            for file_name in file_names:
                if file_name.split(".")[-1] in ["pts"]:
                    pts_file_list.append(os.path.join(file_path, file_name))

    # Show the image one by one.
    total_data_num = len(pts_file_list)
    images_index = 0
    while 1:
       # print("The %sth graph are currently processed." % images_index)
        if (images_index % 100 == 0):
            sys.stdout.write('\r>> Processing image %d/%d' % (images_index + 1, total_data_num))
            sys.stdout.flush()
        pts_file_name = pts_file_list[images_index]
        landmark_points = []
        # Read the points from file.
        landmark_points = read_points_from_pts(pts_file_name)
        # Safe guard, make sure point importing goes well.
        assert len(landmark_points) == LANDMARKS_NUM, "The landmarks should contain %d points." % LANDMARKS_NUM

        write_pts_to_json(pts_file_name, landmark_points, TARGET_LANDMARKS_DIR, TARGET_BOX_DIR, TARGET_IMAGE_DIR)
        images_index += 1

        ch = cv2.waitKey(1)
        if images_index >= len(pts_file_list):
            ch = 27
        # exit
        if ch == 27:
            break
    f_txt_file = os.path.join(OUTPUT_DIR, "raw_count.txt")
    f_txt = open(f_txt_file, 'w')
    f_txt.write("the number of total data: %d\n" % images_index)
    f_txt.close()
    print("\n%d images data are converted !" % images_index)

if __name__ == "__main__":
    INPUT_LIST = ["../data/out"]
    OUTPUT_DIR = "../data/json"
    LANDMARKS_NUM = 68
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    assert os.path.exists(OUTPUT_DIR)
    TARGET_LANDMARKS_DIR = os.path.join(OUTPUT_DIR, "landmark")
    TARGET_BOX_DIR = os.path.join(OUTPUT_DIR, "box")
    TARGET_IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
    if not os.path.exists(TARGET_LANDMARKS_DIR):
        os.mkdir(TARGET_LANDMARKS_DIR)
    if not os.path.exists(TARGET_BOX_DIR):
        os.mkdir(TARGET_BOX_DIR)
    if not os.path.exists(TARGET_IMAGE_DIR):
        os.mkdir(TARGET_IMAGE_DIR)

    main()
