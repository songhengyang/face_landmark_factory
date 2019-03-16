#!/usr/bin/env bash
current_dir=$(pwd);
cd ~/tensorflow-r1.13 && \
bazel run --config=opt \
  //tensorflow/lite/toco:toco -- \
  --input_file=${current_dir}"/../model/facial_landmark_MobileNet.pb" \
  --output_file=${current_dir}"/../model/facial_landmark_MobileNet.tflite" \
  --inference_type=FLOAT \
  --input_shape=1,64,64,1 \
  --input_array=input_2 \
  --output_arrays=output/BiasAdd
