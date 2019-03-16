import glob
import cv2
import numpy as np
import os
import json
#import dlib

def read_points(pts_path):
    with open(pts_path) as file:
        landmarks = []
        for line in file:
            if "version" in line or "points" in line or "{" in line or "}" in line:
                continue
            else:
                x, y = line.strip().split(" ")
                landmarks.append([float(x), float(y)])
        landmarks = np.array(landmarks)
    return landmarks

def get_paths(dir, dataset_name):
    if dataset_name == "300W":
        img_paths = glob.glob(dir + dataset_name + "/*.png")
    elif dataset_name == "lfpw1":
        img_paths = glob.glob(dir + dataset_name + "/*.png")
    elif dataset_name == "lfpw2":
        img_paths = glob.glob(dir + dataset_name + "/*.png")
    elif dataset_name == "300VW":
        img_paths = glob.glob(dir + dataset_name + "/*/annot/*.jpg")
    else:
        img_paths = glob.glob(dir + dataset_name + "/*.jpg")
    return img_paths
'''
def detect_face_rect_of_pts(img_path,pts):
    detector = dlib.get_frontal_face_detector()
   # detect face using dlib
    im = cv2.imread(img_path)
    dets = detector(im, 1)
    selected_det = None
    pts_center = np.mean(pts,axis=0)
    for det in dets:
        if pts_center[0]>det.left() and pts_center[0]<det.right() and\
        pts_center[1]>det.top() and pts_center[1]<det.bottom():
            selected_det = det
    if selected_det is None:
        return None
    left = selected_det.left()
    top = selected_det.top()
    width = selected_det.width()
    height = selected_det.height()
    return [left,top,width,height]
'''