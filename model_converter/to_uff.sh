#!/usr/bin/env bash
/home/jerry/anaconda3/envs/tf13py36/bin/python /home/jerry/TensorRT-5.0.2.6/uff/uff-0.5.5-py2.py3-none-any/uff/bin/convert_to_uff.py \
../model/facial_landmark_SqueezeNet.pb \
--output ../model/facial_landmark_SqueezeNet.uff \
--output-nodes output/BiasAdd
