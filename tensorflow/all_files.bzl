# List of data dependencies for :all_opensource_files
def tf_all_files():
  return [
        ":all_files",
        "//tensorflow/cc:all_files",
        "//tensorflow/contrib:all_files",
        "//tensorflow/contrib/ctc:all_files",
        "//tensorflow/contrib/distributions:all_files",
        "//tensorflow/contrib/framework:all_files",
        "//tensorflow/contrib/linear_optimizer:all_files",
        "//tensorflow/contrib/linear_optimizer/kernels:all_files",
        "//tensorflow/contrib/lookup:all_files",
        "//tensorflow/contrib/losses:all_files",
        "//tensorflow/contrib/layers:all_files",
        "//tensorflow/contrib/skflow:all_files",
        "//tensorflow/contrib/testing:all_files",
        "//tensorflow/contrib/util:all_files",
        "//tensorflow/core:all_files",
        "//tensorflow/core/distributed_runtime:all_files",
        "//tensorflow/core/distributed_runtime/rpc:all_files",
        "//tensorflow/core/kernels:all_files",
        "//tensorflow/core/ops/compat:all_files",
        "//tensorflow/core/platform/default/build_config:all_files",
        "//tensorflow/core/util/ctc:all_files",
        "//tensorflow/examples/android:all_files",
        "//tensorflow/examples/how_tos/reading_data:all_files",
        "//tensorflow/examples/image_retraining:all_files",
        "//tensorflow/examples/label_image:all_files",
        "//tensorflow/examples/tutorials/mnist:all_files",
        "//tensorflow/examples/tutorials/word2vec:all_files",
        "//tensorflow/g3doc/how_tos/adding_an_op:all_files",
        "//tensorflow/g3doc/tutorials:all_files",
        "//tensorflow/models/embedding:all_files",
        "//tensorflow/models/image/alexnet:all_files",
        "//tensorflow/models/image/cifar10:all_files",
        "//tensorflow/models/image/imagenet:all_files",
        "//tensorflow/models/image/mnist:all_files",
        "//tensorflow/models/rnn:all_files",
        "//tensorflow/models/rnn/ptb:all_files",
        "//tensorflow/models/rnn/translate:all_files",
        "//tensorflow/python:all_files",
        "//tensorflow/python/tools:all_files",
        "//tensorflow/tensorboard:all_files",
        "//tensorflow/tensorboard/app:all_files",
        "//tensorflow/tensorboard/backend:all_files",
        "//tensorflow/tensorboard/components:all_files",
        "//tensorflow/tensorboard/lib:all_files",
        "//tensorflow/tensorboard/scripts:all_files",
        "//tensorflow/tools/docker:all_files",
        "//tensorflow/tools/docker/notebooks:all_files",
        "//tensorflow/tools/docs:all_files",
        "//tensorflow/tools/test:all_files",
        "//tensorflow/user_ops:all_files",
    ]
