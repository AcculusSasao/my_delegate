#!/bin/sh
TFROOT=../../../../../
cd $TFROOT
bazel build tensorflow/lite/delegates/utils/my_delegate:my_external_delegate.so
