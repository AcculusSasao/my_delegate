cc_library(
    name = "my_delegate",
    srcs = [
        "my_delegate.cc",
    ],
    hdrs = [
        "my_delegate.h",
    ],
    deps = [
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/tools:logging",
    ],
)
cc_binary(
    name = "my_external_delegate.so",
    srcs = [
        "external_delegate_adaptor.cc",
    ],
    defines = ["TFL_EXTERNAL_DELEGATE_COMPILE_LIBRARY"],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":my_delegate",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/external:external_delegate_interface",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
    ],
)
