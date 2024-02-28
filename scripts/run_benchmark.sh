#!/bin/sh

if [ $# -ne 1 ]; then
	echo "specify tflite model as arg1"
	exit
fi
TFMODEL=$PWD/$1

TFROOT=../../../../../
cd $TFROOT
bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
	--graph=$TFMODEL \
	--external_delegate_path=bazel-bin/tensorflow/lite/delegates/utils/my_delegate/my_external_delegate.so \
	--external_delegate_options='param_a:2;param_b:3'
