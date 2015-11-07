# Description:
<<<<<<< HEAD
#   TensorFlow camera demo app for Android.

load("@build_bazel_rules_android//android:rules.bzl", "android_binary")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

LINKER_SCRIPT = "jni/version_script.lds"

# libtensorflow_demo.so contains the native code for image colorspace conversion
# and object tracking used by the demo. It does not require TF as a dependency
# to build if STANDALONE_DEMO_LIB is defined.
# TF support for the demo is provided separately by libtensorflow_inference.so.
cc_binary(
    name = "libtensorflow_demo.so",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]),
    copts = tf_copts(),
    defines = ["STANDALONE_DEMO_LIB"],
    linkopts = [
        "-landroid",
        "-ldl",
        "-ljnigraphics",
        "-llog",
        "-lm",
        "-z defs",
        "-s",
        "-Wl,--version-script,$(location {})".format(LINKER_SCRIPT),
    ],
    linkshared = 1,
    linkstatic = 1,
=======
#   Tensorflow camera demo app for Android.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "tensorflow_native_libs",
    srcs = glob(["jni/**/*.cc"]),
    hdrs = glob(["jni/**/*.h"]),
    copts = [
        "-std=c++11",
        "-mfpu=neon",
    ],
    linkopts = ["-llog -landroid -lm -ljnigraphics"],
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    tags = [
        "manual",
        "notap",
    ],
    deps = [
<<<<<<< HEAD
        LINKER_SCRIPT,
    ],
)

cc_library(
    name = "tensorflow_native_libs",
    srcs = [
        ":libtensorflow_demo.so",
        "//tensorflow/tools/android/inference_interface:libtensorflow_inference.so",
    ],
    tags = [
        "manual",
        "notap",
    ],
=======
        ":dummy_pthread",
        "//tensorflow/core:android_tensorflow_lib",
    ],
)

# This library only exists as a workaround to satisfy dependencies
# that declare -lpthread in their linkopts. Although Android supports
# pthreads, it does not provide it as a separate library.
cc_library(
    name = "dummy_pthread",
    srcs = ["jni/libpthread.so"],
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
)

android_binary(
    name = "tensorflow_demo",
    srcs = glob([
        "src/**/*.java",
    ]),
<<<<<<< HEAD
    # Package assets from assets dir as well as all model targets. Remove undesired models
    # (and corresponding Activities in source) to reduce APK size.
    assets = [
        "//tensorflow/examples/android/assets:asset_files",
        ":external_assets",
    ],
    assets_dir = "",
    custom_package = "org.tensorflow.demo",
    inline_constants = 1,
=======
    assets = glob(["assets/**"]),
    assets_dir = "assets",
    custom_package = "org.tensorflow.demo",
    inline_constants = 1,
    legacy_native_support = 0,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":tensorflow_native_libs",
<<<<<<< HEAD
        "//tensorflow/tools/android/inference_interface:android_tensorflow_inference_java",
    ],
)

# LINT.IfChange
filegroup(
    name = "external_assets",
    srcs = [
        "@inception_v1//:model_files",
        "@mobile_ssd//:model_files",
        "@speech_commands//:model_files",
        "@stylize//:model_files",
    ],
)
# LINT.ThenChange(//tensorflow/examples/android/download-models.gradle)

filegroup(
    name = "java_files",
    srcs = glob(["src/**/*.java"]),
)

filegroup(
    name = "jni_files",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]),
)

filegroup(
    name = "resource_files",
    srcs = glob(["res/**"]),
)

exports_files([
    "AndroidManifest.xml",
])
=======
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "bin/**",
            "gen/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
