<<<<<<< HEAD
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    deps = [
        "@bazel_skylib//lib:selects",
    ],
=======
licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

load("/tensorflow/tensorflow", "if_cuda")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "cuda_crosstool_condition",
    values = {"crosstool_top": "//third_party/gpus/crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_gcudacc",
    values = {
        "crosstool_top": "//third_party/gpus/crosstool",
        "copt": "--use_gcudacc",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_nvcc",
    values = {
        "crosstool_top": "//third_party/gpus/crosstool",
        "copt": "--use_nvcc",
    },
)

cc_library(
    name = "cuda_headers",
    hdrs = glob([
        "**/*.h",
    ]),
    includes = [".", "include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudart_static",
    srcs = [
        "lib64/libcudart_static.a",
    ],
    includes = ["include/"],
    linkopts = [
        "-ldl",
        "-lrt",
        "-lpthread",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudart",
    srcs = [
        "lib64/libcudart.so.7.0",
    ],
    data = [
        "lib64/libcudart.so.7.0",
    ],
    includes = ["include/"],
    visibility = ["//visibility:public"],
    linkstatic = 1,
)

cc_library(
    name = "cublas",
    srcs = [
        "lib64/libcublas.so.7.0",
    ],
    data = [
        "lib64/libcublas.so.7.0",
    ],
    includes = ["include/"],
    visibility = ["//visibility:public"],
    linkstatic = 1,
)

cc_library(
    name = "cudnn",
    srcs = [
        "lib64/libcudnn.so.6.5",
    ],
    data = [
        "lib64/libcudnn.so.6.5",
    ],
    includes = ["include/"],
    visibility = ["//visibility:public"],
    linkstatic = 1,
)

cc_library(
    name = "cuda",
    deps = [
        ":cuda_headers",
        ":cudart",
        ":cublas",
        ":cudnn",
    ],
    visibility = ["//visibility:public"],
)

# TODO(opensource): for now, we have to invoke the cuda_config.sh manually in the source tree.
# This rule checks if Cuda libraries in the source tree has been properly configured.
# The output list makes bazel runs this rule first if the Cuda files are missing.
# This gives us an opportunity to check and print a meaningful error message.
# But we will need to create the output file list to make bazel happy in a successfull run.
genrule(
    name = "cuda_check",
    srcs = [
        "cuda.config",
        "cuda_config.sh",
    ],
    outs = [
        "include/cuda.h",
        "include/cublas.h",
        "include/cudnn.h",
        "lib64/libcudart_static.a",
        "lib64/libcublas.so.7.0",
        "lib64/libcudnn.so.6.5",
        "lib64/libcudart.so.7.0",
    ],
    cmd = if_cuda(
        # Under cuda config, create all the symbolic links to the actual cuda files
        "OUTPUTDIR=`readlink -f $(@D)/../../..`; cd third_party/gpus/cuda; OUTPUTDIR=$$OUTPUTDIR ./cuda_config.sh --check;",

        # Under non-cuda config, create all dummy files to make the build go through
        ";".join([
          "mkdir -p $(@D)/include",
         "mkdir -p $(@D)/lib64",
         "touch $(@D)/include/cuda.h",
          "touch $(@D)/include/cublas.h",
          "touch $(@D)/include/cudnn.h",
          "touch $(@D)/lib64/libcudart_static.a",
          "touch $(@D)/lib64/libcublas.so.7.0",
          "touch $(@D)/lib64/libcudnn.so.6.5",
          "touch $(@D)/lib64/libcudart.so.7.0"
            ]),
    ),
    local = 1,
)

genrule(
    name = "cuda_config_check",
    outs = [
        "cuda.config",
    ],
    cmd = if_cuda(
        # Under cuda config, create the symbolic link to the actual cuda.config
        "ln -sf `readlink -f third_party/gpus/cuda/cuda.config` $(@D)/;",

        # Under non-cuda config, create the dummy file
        ";".join([
         "touch $(@D)/cuda.config",
        ]),
    ),
    local = 1,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
)
