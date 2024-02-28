#!/bin/sh

TFROOT=../../../../../

python infer.py -i panda.jpg \
	-m mobilenet_v1_0.25_128.tflite \
	-d $TFROOT/bazel-bin/tensorflow/lite/delegates/utils/my_delegate/my_external_delegate.so \
	-o "param_a:2;param_b:3"
