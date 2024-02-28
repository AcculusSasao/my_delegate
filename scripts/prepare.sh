#!/bin/sh

#sudo apt install bazel-bootstrap bazel-6.1.0

TFROOT=../../../../../
cd $TFROOT
bazel build tensorflow/lite:libtensorflowlite.so
bazel build tensorflow/lite/tools/benchmark:benchmark_model
cd -

# download model
#  https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
mkdir tmp; cd tmp
wget http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128.tgz
tar xf mobilenet_v1_0.25_128.tgz
cp mobilenet_v1_0.25_128.tflite ..
cd -
rm -rf tmp

# download test image
wget https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG -O panda.jpg
