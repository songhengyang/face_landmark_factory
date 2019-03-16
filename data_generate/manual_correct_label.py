"""
This script shows how to read iBUG pts file and draw all the landmark points on image.
"""
import json
import os
import numpy as np
import cv2

import data_generate.face_detector as fd
from data_generate.BBox_utils import read_points_from_json, read_points_from_pts
from data_generate.Landmark_utils import points_are_valid, get_valid_box, \
                                         get_minimal_box, move_landmark_point, \
                                         get_location_points


def draw_landmark_indexpoint(image, points, point_index):
    for i in range(0, len(points)):
        point = points[i]
        if i == point_index:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.circle(image, (int(point[0]), int(
            point[1])), 3, color, -1, cv2.LINE_AA)


def preview(point_file, raw_points, taget_landmarks_dir, taget_box_dir, point_index, point_dev):
    new_points = raw_points

    # Read the image.
    head, tail = os.path.split(point_file)
    image_file = tail.split('.')[-2]
    img_jpg = os.path.join(TARGET_IMAGE_DIR, image_file + ".jpg")
    img = cv2.imread(img_jpg)

    # Fast check: all points are in image.
    if points_are_valid(new_points, img) is False:
        print("Invalid json file, ignored:", point_file)
        return None

    # Get the valid facebox.
    facebox = get_valid_box(img, new_points)
    box_color = (255, 0, 0)
    if facebox is None:
        print("Using minimal box.")
        facebox = get_minimal_box(new_points)
        box_color = (0, 0, 255)
    fd.draw_box(img, [facebox], box_color)

    new_local_points = get_location_points(facebox, new_points)

    # Extract valid image area.
    face_area = img[facebox[1]:facebox[3],
                    facebox[0]: facebox[2]].copy()

    # Resize image and compute local points.
    width = facebox[2] - facebox[0]
    height = facebox[3] - facebox[1]
    if width != height:
        print('opps!', width, height)
    dis_W = 512
    dis_H = 512
    zoom_rate = 1.0
    if width*height != dis_W*dis_H:
        zoom_rate = dis_W / width
        if width != height:
            dis_H = int(height * zoom_rate)
        face_area = cv2.resize(face_area, (dis_W, dis_H))
        for i in range(0, len(new_local_points)):
            for j in range(0, len(new_local_points[i])):
                new_local_points[i][j] = float(new_local_points[i][j]*zoom_rate)

    # move index landmark
    move_landmark_point(new_local_points, point_index, point_dev)
    for i in range(0, len(point_dev)):
        new_points[point_index][i] = new_local_points[point_index][i]/zoom_rate + facebox[i]

    # New file to be written.
    head, tail = os.path.split(point_file)
    subset_name = head.split('/')[-2]
    common_file_name = tail.split('.')[-2]
    landmark_common_url = os.path.join(
        taget_landmarks_dir, common_file_name)
    box_common_url = os.path.join(
        taget_box_dir, common_file_name)

    # Save the landmarks location.
    landmark_json_url = landmark_common_url + ".json"
    points_to_save = np.array(new_points).flatten()
    with open(landmark_json_url, mode='w') as file1:
        json.dump(list(points_to_save), file1)

    # Save the box location.
    box_json_url = box_common_url + ".json"
    with open(box_json_url, mode='w') as file2:
        json.dump(facebox, file2)

    print("New file saved:", landmark_json_url, box_json_url, sep='\n')

    # Show the result.
    draw_landmark_indexpoint(face_area, new_local_points, point_index)
    cv2.imshow("resized face", face_area)

    draw_landmark_indexpoint(img, new_points, point_index)
    # Show whole image in window.
    width, height = img.shape[:2]
    max_height = 640
    if height > max_height:
        img = cv2.resize(
            img, (max_height, int(width * max_height / height)))
    cv2.imshow("raw image", img)



def main():
    # List existent json files
    json_file_list = []
    for file_path, _, file_names in os.walk(TARGET_LANDMARKS_DIR):
        for file_name in file_names:
            if file_name.split(".")[-1] in ["json"]:
                json_file_list.append(os.path.join(file_path, file_name))

    # Show the image one by one.
    images_index = 0
    images_index_old = images_index
    points_index = 0
    point_dev = [0, 0]
    while 1:
        if images_index != images_index_old:
            if images_index < 0:
                images_index = len(json_file_list) + images_index
            if images_index > len(json_file_list)-1:
                images_index = 0
            points_index = 0
            point_dev[0] = 0
            point_dev[1] = 0
        print("The %sth graph are currently processed." % images_index)
        json_file_name = json_file_list[images_index]

        # Read the points from file.
        landmark_points = read_points_from_json(json_file_name)
        # Safe guard, make sure point importing goes well.
        assert len(landmark_points) == POINTS_NUM, "The landmarks should contain %d points." % POINTS_NUM

        preview(json_file_name, landmark_points, TARGET_LANDMARKS_DIR, TARGET_BOX_DIR, points_index, point_dev)

        images_index_old = images_index
        ch = cv2.waitKey(0)
        # exit
        if ch == 27:
            break
        # next image (space)
        elif ch == 32:
            images_index += 1
        # last image
        elif ch == ord('b'):
            images_index -= 1
        # last point in landmarks
        elif ch == ord('q'):
            if points_index > 0:
                points_index -= 1
            else:
                points_index = POINTS_NUM - 1
        # next point in landmarks
        elif ch == ord('e'):
            if points_index < POINTS_NUM - 1:
                points_index += 1
            else:
                points_index = 0
        elif ch == ord('a'):
            point_dev[0] = -1
            point_dev[1] = 0
        elif ch == ord('d'):
            point_dev[0] = 1
            point_dev[1] = 0
        elif ch == ord('w'):
            point_dev[0] = 0
            point_dev[1] = -1
        elif ch == ord('s'):
            point_dev[0] = 0
            point_dev[1] = 1
        else:
            point_dev[0] = 0
            point_dev[1] = 0


if __name__ == "__main__":
    POINTS_NUM = 68
    INPUT_DIR = "../data/json"
    TARGET_LANDMARKS_DIR = os.path.join(INPUT_DIR, "landmark")
    TARGET_BOX_DIR = os.path.join(INPUT_DIR, "box")
    TARGET_IMAGE_DIR = os.path.join(INPUT_DIR, "images")
    if not os.path.exists(TARGET_LANDMARKS_DIR):
        os.mkdir(TARGET_LANDMARKS_DIR)
    if not os.path.exists(TARGET_BOX_DIR):
        os.mkdir(TARGET_BOX_DIR)
    if not os.path.exists(TARGET_IMAGE_DIR):
        os.mkdir(TARGET_IMAGE_DIR)

    main()
