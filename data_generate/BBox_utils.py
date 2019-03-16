# coding: utf-8

import os
import json
from os.path import join, exists
import time
import cv2
import numpy as np
import sys


def logger(msg):
    """
        log message
    """
    now = time.ctime()
    print("[%s] %s" % (now, msg))

def createDir(p):
    if not os.path.exists(p):
        os.mkdir(p)
#shuffle in the same way
def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def drawLandmarkBox(img, bbox, landmark):
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
    return img

def drawLandmark(image, landmark, color=(255, 255, 255), thick=1):
    """Draw mark points on image"""
    for x, y in landmark:
        cv2.circle(image, (int(x), int(y)), thick, color, -1, cv2.LINE_AA)

def read_points_from_json(file_name=None):

    data = []
    with open(file_name) as json_file:
        data = json.load(json_file)
    points = []
    for i in range(0, len(data), 2):
        points.append([float(data[i]), float(data[i+1])])
    return points

def read_points_from_pts(file_name=None):
    """
    Read points from .pts file.
    """
    points = []
    with open(file_name) as file:
        line_count = 0
        for line in file:
            if "version" in line or "points" in line or "{" in line or "}" in line:
                continue
            else:
                loc_x, loc_y = line.strip().split(sep=" ")
                points.append([float(loc_x), float(loc_y)])
                line_count += 1
    return points

def getDataFromTxt(txt,data_path, with_landmark=True):
    """
        Generate data from txt file
        return [(img_path, bbox, landmark)]
            bbox: [left, right, top, bottom]
            landmark: [(x1, y1), (x2, y2), ...]
    """


    with open(txt, 'r') as fd:
        lines = fd.readlines()

    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        img_path = os.path.join(data_path, components[0]).replace('\\','/') # file path

        # bounding box, (x1, y1, x2, y2)
        #bbox = (components[1], components[2], components[3], components[4])
        bbox = (components[1], components[3], components[2], components[4])        
        bbox = [float(_) for _ in bbox]
        bbox = list(map(int,bbox))
        # landmark
        if not with_landmark:
            result.append((img_path, BBox(bbox)))
            continue
        landmark = np.zeros((5, 2))
        for index in range(0, 5):
            rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = rv
        #normalize
        '''
        for index, one in enumerate(landmark):
            rv = ((one[0]-bbox[0])/(bbox[2]-bbox[0]), (one[1]-bbox[1])/(bbox[3]-bbox[1]))
            landmark[index] = rv
        '''
        result.append((img_path, BBox(bbox), landmark))
    return result

def getDataFromJson(image_dir, box_dir, landmark_dir, landmarks_num):
    # List all the json files.
    box_file_list = []
    for file_path, _, file_names in os.walk(box_dir):
        for file_name in file_names:
            if file_name.split(".")[-1] in ["json"]:
                box_file_list.append(os.path.join(file_path, file_name))

    result = []
    i = 0
    file_num = len(box_file_list)
    for json_file_i in box_file_list:
        head_0, tail_0 = os.path.split(json_file_i)
        img_name = tail_0.split(".")[-2] + ".jpg"
        img_path = os.path.join(image_dir, img_name).replace('\\', '/')
        # Read points and image, make sure point importing goes well.
        landmark_name = tail_0.split(".")[-2] + ".json"
        landmark_file = os.path.join(landmark_dir, landmark_name).replace('\\', '/')
        landmarks = read_points_from_json(landmark_file)
        # Safe guard, make sure point importing goes well.
        assert len(landmarks) == landmarks_num, "The landmarks should contain %s points." % landmarks_num
        landmark_np = np.zeros((landmarks_num, 2))
        for index in range(0, landmarks_num):
            rv = (float(landmarks[index][0]), float(landmarks[index][1]))
            landmark_np[index] = rv

        head_1, tail_1 = os.path.split(json_file_i)
        box_file_name = os.path.join(head_1, tail_0)
        box = read_points_from_json(box_file_name)

        bbox = (box[0][0], box[0][1], box[1][0], box[1][1])
        bbox = [float(_) for _ in bbox]
        bbox = list(map(int, bbox))

        result.append((img_path, BBox(bbox), landmark_np))
        i += 1
        if (i%100 == 0):
            sys.stdout.write('\r>> read json file %d/%d' % (i, file_num))
            sys.stdout.flush()
    return result


def read_CelebA_data(box_txt, landmark_txt, image_path, landmarks_num):
    """
        Generate data from txt file
        return [(img_path, bbox, landmark)]
            bbox: [left, right, top, bottom]
            landmark: [(x1, y1), (x2, y2), ...]
    """


    with open(box_txt, 'r') as f_box:
        lines_box = f_box.readlines()
    with open(landmark_txt, 'r') as fd_landmark:
        lines_landmark = fd_landmark.readlines()

    result = []
    line_num = 0
    for line in zip(lines_box, lines_landmark):
        line_num += 1
        if line_num <= 2:
            continue
        components_0 = line[0].split()
        components_1 = line[1].split()
        img_path = os.path.join(image_path, components_0[0]).replace('\\','/') # file path

        width = float(components_0[3])
        height = float(components_0[4])
        # bounding box, (x1, y1, x2, y2)
        bbox = (components_0[1], components_0[2], float(components_0[1])+width, float(components_0[2])+height)
      #  bbox = (components[1], components[3], components[2], components[4])
        bbox = [float(_) for _ in bbox]
        bbox = list(map(int,bbox))
        # landmark
        landmark = np.zeros((landmarks_num, 2))
        for index in range(0, landmarks_num):
            rv = (float(components_1[1+2*index]), float(components_1[2+2*index]))
            landmark[index] = rv
        #normalize
        '''
        for index, one in enumerate(landmark):
            rv = ((one[0]-bbox[0])/(bbox[2]-bbox[0]), (one[1]-bbox[1])/(bbox[3]-bbox[1]))
            landmark[index] = rv
        '''
        result.append((img_path, BBox(bbox), landmark))
    return result


def getPatch(img, bbox, point, padding):
    """
        Get a patch iamge around the given point in bbox with padding
        point: relative_point in [0, 1] in bbox
    """
    point_x = bbox.x + point[0] * bbox.w
    point_y = bbox.y + point[1] * bbox.h
    patch_left = point_x - bbox.w * padding
    patch_right = point_x + bbox.w * padding
    patch_top = point_y - bbox.h * padding
    patch_bottom = point_y + bbox.h * padding
    patch = img[patch_top: patch_bottom+1, patch_left: patch_right+1]
    patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])
    return patch, patch_bbox

'''
def processImageOriginal(imgs):
    """
        process images before feeding to CNNs
        imgs: N x 1 x W x H
    """
    imgs = imgs.astype(np.float32)
    for i, img in enumerate(imgs):
        m = img.mean()
        s = img.std()
        imgs[i] = (img - m) / s
    return imgs
'''
def processImage(imgs):
    """
        process images before feeding to CNNs
        imgs: N x 1 x W x H
    """
    imgs = imgs.astype(np.float32)
    for i, img in enumerate(imgs):
        imgs[i] = (img - 127.5) / 128
    return imgs

def dataArgument(data):
    """
        dataArguments
        data:
            imgs: N x 1 x W x H
            bbox: N x BBox
            landmarks: N x 10
    """
    pass

class BBox(object):
    """
        Bounding Box of face
    """
    def __init__(self, bbox):
        self.left = bbox[0]
        self.top = bbox[1]
        self.right = bbox[2]
        self.bottom = bbox[3]
        
        self.x = bbox[0]
        self.y = bbox[1]
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)
    #offset
    def project(self, point):
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x, y])
    #absolute position(image (left,top))
    def reproject(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])
    #landmark: 5*2
    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p
    #change to offset according to bbox
    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p
    #f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
    #self.w bounding-box width
    #self.h bounding-box height
    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])
