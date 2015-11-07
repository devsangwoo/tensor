<<<<<<< HEAD
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functional tests for pooling operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def GetDeviceScope(self, use_gpu=False):
  if context.executing_eagerly():
    if use_gpu and test.is_gpu_available():
      return ops.device("GPU:0")
    return ops.device("CPU:0")
  else:
    return self.session(use_gpu=use_gpu)


def GetTestConfigs(include_nchw_vect_c=False):
  """Get all the valid tests configs to run.

  Args:
    include_nchw_vect_c: Whether to include NCHW_VECT_C in the test configs.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  test_configs = [("NHWC", False), ("NHWC", True)]
  if not test.is_gpu_available(cuda_only=True):
    tf_logging.info("NCHW and NCHW_VECT_C tests skipped because not run with "
                    "--config=cuda or no GPUs available.")
    return test_configs
  # "NCHW" format is currently supported exclusively on CUDA GPUs.
  test_configs += [("NCHW", True)]
  if include_nchw_vect_c:
    if test.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(6, 1)):
      test_configs += [("NCHW_VECT_C", True)]
    else:
      tf_logging.info("NCHW_VECT_C test skipped because no GPUs with "
                      "compute capability >= 6.1 are available.")

  return test_configs


def GetShrunkInceptionMaxPoolShapes(shrink=30):
  """Iterator for some of the max pool ops in the Inception 2015 model.

  Args:
    shrink: Factor to shrink depth relative to Inception.

=======
"""Functional tests for pooling operations."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker as gc
from tensorflow.python.ops import gen_nn_ops


def GetInceptionMaxPoolShapes():
  """Iterator for some of the max pool ops in the Inception 2015 model.

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  Yields:
    Tuple (name, input_size, filter_size, out_size, strides, padding)
  """
  names = ["maxpool2", "maxpool3", "maxpool4", "maxpool5"]
<<<<<<< HEAD
  input_sizes = [[32, 71, 71, 192], [32, 35, 35, 288], [32, 17, 17, 1248],
                 [32, 8, 8, 2048]]
  filter_sizes = [[1, 3, 3, 1], [1, 3, 3, 1], [1, 3, 3, 1], [1, 3, 3, 1]]
  output_sizes = [[32, 35, 35, 192], [32, 17, 17, 288], [32, 8, 8, 1248],
                  [32, 8, 8, 2048]]
  strides = [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]]
  # Shrink each depth value
  for i in input_sizes:
    i[3] //= shrink
  for o in output_sizes:
    o[3] //= shrink
=======
  input_sizes = [[32, 71, 71, 192],
                 [32, 35, 35, 288], [32, 17, 17, 1248], [32, 8, 8, 2048]]
  filter_sizes = [[1, 3, 3, 1], [1, 3, 3, 1],
                  [1, 3, 3, 1], [1, 3, 3, 1]]
  output_sizes = [[32, 35, 35, 192], [32, 17, 17, 288],
                  [32, 8, 8, 1248], [32, 8, 8, 2048]]
  strides = [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1],
             [1, 1, 1, 1]]
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  paddings = ["VALID", "VALID", "VALID", "SAME"]
  for n, i, f, o, s, p in zip(names, input_sizes, filter_sizes, output_sizes,
                              strides, paddings):
    yield n, i, f, o, s, p


<<<<<<< HEAD
class PoolingTest(test.TestCase):

  def _VerifyOneType(self, pool_func, input_sizes, ksize, strides, padding,
                     data_format, data_type, expected, use_gpu, v2):
=======
class PoolingTest(tf.test.TestCase):

  def _VerifyValues(self, pool_func, input_sizes, ksize, strides, padding,
                    expected, use_gpu):
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
<<<<<<< HEAD
      data_format: The data format we use to run the pooling operation.
      data_type: The data type to use to run the pooling operation.
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether we are running on GPU.
    """
    total_size = 1
    for s in input_sizes:
      total_size *= s
<<<<<<< HEAD
    if v2 and data_format != "NHWC":
      tf_logging.info("v2 not supported for %s", data_format)
      return
    if data_format == "NCHW_VECT_C":
      if data_type != dtypes.float32:
        tf_logging.info("quantization to qint8 not implemented for %r",
                        data_type)
        return
      if input_sizes[-1] % 4 != 0:
        tf_logging.info("Skipping test for depth %d", input_sizes[-1])
        return
    tf_logging.info("Running %s test. %r %r %d %r %r %r %s", data_format, v2,
                    input_sizes, total_size, pool_func, ksize, strides,
                    data_type)
    # Initializes the input tensor with array containing incrementing
    # numbers from 1, wrapping round to -127 after 127 to support int8.
    x = [((f + 128) % 255) - 127 for f in range(total_size)]
    with self.cached_session(use_gpu=use_gpu):
      t = constant_op.constant(x, shape=input_sizes, dtype=data_type)
      if data_format in ("NCHW", "NCHW_VECT_C"):
        if data_format == "NCHW_VECT_C":
          t = test_util.NHWCToNCHW_VECT_C(t)
          t, _, _ = gen_array_ops.quantize_v2(t, -128.0, 127.0, dtypes.qint8)
        else:
          t = test_util.NHWCToNCHW(t)
        ksize = test_util.NHWCToNCHW(ksize)
        strides = test_util.NHWCToNCHW(strides)
      ksize_placeholder = array_ops.placeholder(dtypes.int32, shape=[4])
      strides_placeholder = array_ops.placeholder(dtypes.int32, shape=[4])
      if v2:
        t = pool_func(
            t,
            ksize=ksize_placeholder,
            strides=strides_placeholder,
            padding=padding,
            data_format=data_format)
      else:
        t = pool_func(
            t,
            ksize=ksize,
            strides=strides,
            padding=padding,
            data_format=data_format)
      if data_format == "NCHW_VECT_C":
        t = gen_array_ops.dequantize(t, -128, 127)
        t = test_util.NCHW_VECT_CToNHWC(t)
      elif data_format == "NCHW":
        t = test_util.NCHWToNHWC(t)
      if v2:
        actual = t.eval(feed_dict={
            ksize_placeholder: ksize,
            strides_placeholder: strides
        })
      else:
        actual = self.evaluate(t)
        self.assertShapeEqual(actual, t)
      self.assertAllCloseAccordingToType(expected, actual.flatten())

  def _VerifyOneTest(self, pool_func, input_sizes, ksize, strides, padding,
                     data_format, expected, use_gpu, v2):
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
      data_format: The data format we use to run the pooling operation.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether we are running on GPU.
    """
    if data_format == "NCHW_VECT_C":
      avg_pool_func = nn_ops.avg_pool
      tf_logging.info("pool_func=%s", pool_func)
      if pool_func == avg_pool_func:
        tf_logging.info("NCHW_VECT_C not yet implemented for avg_pool")
        return

    self._VerifyOneType(pool_func, input_sizes, ksize, strides, padding,
                        data_format, dtypes.float32, expected, use_gpu, v2)
    if not test.is_built_with_rocm():
      # double datatype is not supported for pooling ops on the ROCm platform
      self._VerifyOneType(pool_func, input_sizes, ksize, strides, padding,
                          data_format, dtypes.float64, expected, use_gpu, v2)

    if not use_gpu or test_util.GpuSupportsHalfMatMulAndConv():
      self._VerifyOneType(pool_func, input_sizes, ksize, strides, padding,
                          data_format, dtypes.float16, expected, use_gpu, v2)

  def _VerifyValues(self,
                    pool_func,
                    input_sizes,
                    ksize,
                    strides,
                    padding,
                    expected,
                    use_gpu,
                    v2=False):
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether we are running on GPU.
    """
    for (data_format, use_gpu_2) in GetTestConfigs(True):
      if use_gpu_2 == use_gpu:
        self._VerifyOneTest(pool_func, input_sizes, ksize, strides, padding,
                            data_format, expected, use_gpu, v2)

  def _testAvgPoolValidPadding(self, use_gpu):
    expected_output = [7.0, 8.0, 9.0]
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
        expected=expected_output,
        use_gpu=use_gpu)

  def _testAvgPoolEmpty(self, use_gpu):
    expected_output = [7.0, 8.0, 9.0]
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[1, 3, 3, 0],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
        expected=expected_output,
        use_gpu=use_gpu)

  def _testAvgPoolSamePadding(self, use_gpu):
    expected_output = [8.5, 9.5, 10.5, 14.5, 15.5, 16.5]
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[1, 2, 4, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        use_gpu=use_gpu)
=======
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x = [f * 1.0 for f in range(1, total_size + 1)]
    with self.test_session(use_gpu=use_gpu) as sess:
      t = tf.constant(x, shape=input_sizes)
      t = pool_func(t, ksize=ksize, strides=strides, padding=padding)
      actual = t.eval()
      self.assertAllClose(expected, actual.flatten())
      self.assertShapeEqual(actual, t)

  def _testAvgPoolValidPadding(self, use_gpu):
    expected_output = [7.0, 8.0, 9.0]
    self._VerifyValues(tf.nn.avg_pool, input_sizes=[1, 3, 3, 3],
                       ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding="VALID",
                       expected=expected_output, use_gpu=use_gpu)

  def _testAvgPoolSamePadding(self, use_gpu):
    expected_output = [8.5, 9.5, 10.5, 14.5, 15.5, 16.5]
    self._VerifyValues(tf.nn.avg_pool, input_sizes=[1, 2, 4, 3],
                       ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding="SAME",
                       expected=expected_output, use_gpu=use_gpu)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def _testAvgPoolSamePaddingNonSquareWindow(self, use_gpu):
    # input is:
    # [1.0, 2.0
    #  3.0  4.0]
    #
    # Window of [x, x] should do:
    #  [avg(1.0, 2.0), avg(2.0, padded0),
    #   avg(3.0, 4.0), avg(4.0, padded0)]
<<<<<<< HEAD
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[1, 2, 2, 1],
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[1.5, 2.0, 3.5, 4.0],
        use_gpu=use_gpu)
=======
    self._VerifyValues(tf.nn.avg_pool, input_sizes=[1, 2, 2, 1],
                       ksize=[1, 1, 2, 1], strides=[1, 1, 1, 1],
                       padding="SAME",
                       expected=[1.5, 2.0, 3.5, 4.0], use_gpu=use_gpu)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

    # Window of [x,
    #            x] should do:
    #  [avg(1.0, 3.0), avg(2.0, 4.0)
    #   avg(3.0, padded0), avg(4.0, padded0)]
<<<<<<< HEAD
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[1, 2, 2, 1],
        ksize=[1, 2, 1, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[2.0, 3.0, 3.0, 4.0],
        use_gpu=use_gpu)

  def _testAvgPoolSamePaddingNonSquareWindowMultiBatch(self, use_gpu):
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 2, 2],
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[
            2.0, 3.0, 3.0, 4.0, 6.0, 7.0, 7.0, 8.0, 10.0, 11.0, 11.0, 12.0,
            14.0, 15.0, 15.0, 16.0
        ],
        use_gpu=use_gpu)
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 2, 2],
        ksize=[1, 2, 1, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[
            3.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0,
            13.0, 14.0, 15.0, 16.0
        ],
        use_gpu=use_gpu)

  def _testAvgPoolValidPaddingUnevenStride(self, use_gpu):
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 1, 2, 1],
        padding="VALID",
        expected=[7.0, 8.0, 9.0, 16.0, 17.0, 18.0],
        use_gpu=use_gpu)
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 1, 1],
        padding="VALID",
        expected=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        use_gpu=use_gpu)

  def _testAvgPoolSamePadding4(self, use_gpu):
    expected_output = [
        11.0, 12.0, 13.0, 14.0, 19.0, 20.0, 21.0, 22.0, 43.0, 44.0, 45.0, 46.0,
        51.0, 52.0, 53.0, 54.0
    ]
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[1, 4, 4, 4],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        use_gpu=use_gpu)

  def _testAvgPoolSamePaddingPacket4(self, use_gpu):
    expected_output = [
        21.0, 22.0, 23.0, 24.0, 27.0, 28.0, 29.0, 30.0, 45.0, 46.0, 47.0, 48.0,
        51.0, 52.0, 53.0, 54.0
    ]
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[1, 4, 4, 4],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        use_gpu=use_gpu)

  def _testAvgPoolSamePaddingPacket8(self, use_gpu):
    expected_output = [
        -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, 4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
        32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, -3.5, -54.0, -53.0, -52.0,
        -51.0, -50.0, -49.0, -48.0, -47.0, -38.0, -37.0, -36.0, -35.0, -34.0,
        -33.0, -32.0, -31.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0, -16.0,
        -15.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -11.0, -10.0,
        -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        12.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 33.0, 34.0, 35.0,
        36.0, 37.0, 38.0, -3.5, -2.5, -85.0, -84.0, -83.0, -82.0, -81.0, -80.0,
        -79.0, -78.0, -69.0, -68.0, -67.0, -66.0, -65.0, -64.0, -63.0, -62.0,
        -53.0, -52.0, -51.0, -50.0, -49.0, -48.0, -47.0, -46.0, -41.0, -40.0,
        -39.0, -38.0, -37.0, -36.0, -35.0, -34.0
    ]

    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[1, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        use_gpu=use_gpu)

  def _testAvgPoolEmptyInput(self, use_gpu):
    self._VerifyValues(
        nn_ops.avg_pool,
        input_sizes=[0, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[],
        use_gpu=use_gpu)

  @test_util.run_deprecated_v1
=======
    self._VerifyValues(tf.nn.avg_pool, input_sizes=[1, 2, 2, 1],
                       ksize=[1, 2, 1, 1], strides=[1, 1, 1, 1],
                       padding="SAME",
                       expected=[2.0, 3.0, 3.0, 4.0], use_gpu=use_gpu)

  def _testAvgPoolSamePaddingNonSquareWindowMultiBatch(self, use_gpu):
    self._VerifyValues(tf.nn.avg_pool, input_sizes=[2, 2, 2, 2],
                       ksize=[1, 1, 2, 1], strides=[1, 1, 1, 1],
                       padding="SAME",
                       expected=[2.0, 3.0, 3.0, 4.0,
                                 6.0, 7.0, 7.0, 8.0,
                                 10.0, 11.0, 11.0, 12.0,
                                 14.0, 15.0, 15.0, 16.0],
                       use_gpu=use_gpu)
    self._VerifyValues(tf.nn.avg_pool, input_sizes=[2, 2, 2, 2],
                       ksize=[1, 2, 1, 1], strides=[1, 1, 1, 1],
                       padding="SAME",
                       expected=[3.0, 4.0, 5.0, 6.0,
                                 5.0, 6.0, 7.0, 8.0,
                                 11.0, 12.0, 13.0, 14.0,
                                 13.0, 14.0, 15.0, 16.0],
                       use_gpu=use_gpu)

  def _testAvgPoolValidPaddingUnevenStride(self, use_gpu):
    self._VerifyValues(tf.nn.avg_pool, input_sizes=[1, 3, 3, 3],
                       ksize=[1, 2, 2, 1], strides=[1, 1, 2, 1],
                       padding="VALID",
                       expected=[7.0, 8.0, 9.0, 16.0, 17.0, 18.0],
                       use_gpu=use_gpu)
    self._VerifyValues(tf.nn.avg_pool, input_sizes=[1, 3, 3, 3],
                       ksize=[1, 2, 2, 1], strides=[1, 2, 1, 1],
                       padding="VALID",
                       expected=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                       use_gpu=use_gpu)

  def _testAvgPoolSamePadding4(self, use_gpu):
    expected_output = [11.0, 12.0, 13.0, 14.0, 19.0, 20.0, 21.0, 22.0, 43.0,
                       44.0, 45.0, 46.0, 51.0, 52.0, 53.0, 54.0]
    self._VerifyValues(tf.nn.avg_pool, input_sizes=[1, 4, 4, 4],
                       ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding="SAME",
                       expected=expected_output, use_gpu=use_gpu)

  def _testAvgPoolSamePaddingPacket4(self, use_gpu):
    expected_output = [21.0, 22.0, 23.0, 24.0, 27.0, 28.0, 29.0, 30.0,
                       45.0, 46.0, 47.0, 48.0, 51.0, 52.0, 53.0, 54.0]
    self._VerifyValues(tf.nn.avg_pool, input_sizes=[1, 4, 4, 4],
                       ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding="SAME",
                       expected=expected_output, use_gpu=use_gpu)

  def _testAvgPoolSamePaddingPacket8(self, use_gpu):
    expected_output = [73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 89.0,
                       90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 105.0, 106.0,
                       107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 117.0, 118.0,
                       119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 201.0, 202.0,
                       203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 217.0, 218.0,
                       219.0, 220.0, 221.0, 222.0, 223.0, 224.0, 233.0, 234.0,
                       235.0, 236.0, 237.0, 238.0, 239.0, 240.0, 245.0, 246.0,
                       247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 329.0, 330.0,
                       331.0, 332.0, 333.0, 334.0, 335.0, 336.0, 345.0, 346.0,
                       347.0, 348.0, 349.0, 350.0, 351.0, 352.0, 361.0, 362.0,
                       363.0, 364.0, 365.0, 366.0, 367.0, 368.0, 373.0, 374.0,
                       375.0, 376.0, 377.0, 378.0, 379.0, 380.0, 425.0, 426.0,
                       427.0, 428.0, 429.0, 430.0, 431.0, 432.0, 441.0, 442.0,
                       443.0, 444.0, 445.0, 446.0, 447.0, 448.0, 457.0, 458.0,
                       459.0, 460.0, 461.0, 462.0, 463.0, 464.0, 469.0, 470.0,
                       471.0, 472.0, 473.0, 474.0, 475.0, 476.0]
    self._VerifyValues(tf.nn.avg_pool, input_sizes=[1, 8, 8, 8],
                       ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding="SAME",
                       expected=expected_output, use_gpu=use_gpu)

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  def testAvgPooling(self):
    for use_gpu in True, False:
      self._testAvgPoolValidPadding(use_gpu)
      self._testAvgPoolSamePadding(use_gpu)
      self._testAvgPoolSamePaddingNonSquareWindow(use_gpu)
      self._testAvgPoolSamePaddingNonSquareWindowMultiBatch(use_gpu)
      self._testAvgPoolValidPaddingUnevenStride(use_gpu)
      self._testAvgPoolSamePadding4(use_gpu)
      self._testAvgPoolSamePaddingPacket4(use_gpu)
      self._testAvgPoolSamePaddingPacket8(use_gpu)
<<<<<<< HEAD
      self._testAvgPoolEmptyInput(use_gpu)

  def _testMaxPoolValidPadding(self, use_gpu):
    expected_output = [13.0, 14.0, 15.0]
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
        expected=expected_output,
        use_gpu=use_gpu)

    for v2 in [True, False]:
      self._VerifyValues(
          gen_nn_ops.max_pool_v2,
          input_sizes=[1, 3, 3, 3],
          ksize=[1, 2, 2, 1],
          strides=[1, 2, 2, 1],
          padding="VALID",
          expected=expected_output,
          use_gpu=use_gpu,
          v2=v2)

  def _testMaxPoolSamePadding(self, use_gpu):
    expected_output = [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 2, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        use_gpu=use_gpu)

    for v2 in [True, False]:
      self._VerifyValues(
          gen_nn_ops.max_pool_v2,
          input_sizes=[1, 2, 3, 3],
          ksize=[1, 2, 2, 1],
          strides=[1, 2, 2, 1],
          padding="SAME",
          expected=expected_output,
          use_gpu=use_gpu,
          v2=v2)
=======

  def _testMaxPoolValidPadding(self, use_gpu):
    expected_output = [13.0, 14.0, 15.0]
    self._VerifyValues(tf.nn.max_pool, input_sizes=[1, 3, 3, 3],
                       ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding="VALID",
                       expected=expected_output, use_gpu=use_gpu)

  def _testMaxPoolSamePadding(self, use_gpu):
    expected_output = [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    self._VerifyValues(tf.nn.max_pool, input_sizes=[1, 2, 3, 3],
                       ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding="SAME",
                       expected=expected_output, use_gpu=use_gpu)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def _testMaxPoolSamePaddingNonSquareWindow(self, use_gpu):
    # input is:
    # [1.0, 2.0
    #  3.0  4.0]
    #
    # Window of [x, x] should do:
    #
    #  [max(1.0, 2.0), max(2.0, padded0),
    #   max(3.0, 4.0), max(4.0, padded0)]
<<<<<<< HEAD
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 2, 2, 1],
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[2.0, 2.0, 4.0, 4.0],
        use_gpu=use_gpu)

    for v2 in [True, False]:
      self._VerifyValues(
          gen_nn_ops.max_pool_v2,
          input_sizes=[1, 2, 2, 1],
          ksize=[1, 1, 2, 1],
          strides=[1, 1, 1, 1],
          padding="SAME",
          expected=[2.0, 2.0, 4.0, 4.0],
          use_gpu=use_gpu,
          v2=v2)

  def _testMaxPoolValidPaddingUnevenStride(self, use_gpu):
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 1, 2, 1],
        padding="VALID",
        expected=[6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        use_gpu=use_gpu)
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 1, 1],
        padding="VALID",
        expected=[6.0, 7.0, 8.0, 14.0, 15.0, 16.0],
        use_gpu=use_gpu)

    for v2 in [True, False]:
      self._VerifyValues(
          gen_nn_ops.max_pool_v2,
          input_sizes=[1, 4, 4, 1],
          ksize=[1, 2, 2, 1],
          strides=[1, 1, 2, 1],
          padding="VALID",
          expected=[6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
          use_gpu=use_gpu,
          v2=v2)
      self._VerifyValues(
          gen_nn_ops.max_pool_v2,
          input_sizes=[1, 4, 4, 1],
          ksize=[1, 2, 2, 1],
          strides=[1, 2, 1, 1],
          padding="VALID",
          expected=[6.0, 7.0, 8.0, 14.0, 15.0, 16.0],
          use_gpu=use_gpu,
          v2=v2)

  def _testMaxPoolSamePaddingPacket4(self, use_gpu):
    expected_output = [
        21.0, 22.0, 23.0, 24.0, 29.0, 30.0, 31.0, 32.0, 53.0, 54.0, 55.0, 56.0,
        61.0, 62.0, 63.0, 64.0
    ]
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 4, 4, 4],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        use_gpu=use_gpu)

    for v2 in [True, False]:
      self._VerifyValues(
          gen_nn_ops.max_pool_v2,
          input_sizes=[1, 4, 4, 4],
          ksize=[1, 2, 2, 1],
          strides=[1, 2, 2, 1],
          padding="SAME",
          expected=expected_output,
          use_gpu=use_gpu,
          v2=v2)

  def _testMaxPoolSamePaddingPacket8(self, use_gpu):
    expected_output = [
        81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 97.0, 98.0, 99.0, 100.0,
        101.0, 102.0, 103.0, 104.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0,
        119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 120.0,
        18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 34.0, 35.0, 36.0, 37.0,
        38.0, 39.0, 40.0, 41.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
        58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 82.0, 83.0, 84.0, 85.0,
        86.0, 87.0, 88.0, 89.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0,
        105.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0,
        123.0, 124.0, 125.0, 126.0, 127.0, 120.0, 121.0, -45.0, -44.0, -43.0,
        -42.0, -41.0, -40.0, -39.0, -38.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0,
        -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0
    ]
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        use_gpu=use_gpu)

    for v2 in [True, False]:
      self._VerifyValues(
          gen_nn_ops.max_pool_v2,
          input_sizes=[1, 8, 8, 8],
          ksize=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding="SAME",
          expected=expected_output,
          use_gpu=use_gpu,
          v2=v2)

  def _testMaxPoolEmptyInput(self, use_gpu):
    self._VerifyValues(
        gen_nn_ops.max_pool_v2,
        input_sizes=[0, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[],
        use_gpu=use_gpu)

  @test_util.run_deprecated_v1
=======
    self._VerifyValues(tf.nn.max_pool, input_sizes=[1, 2, 2, 1],
                       ksize=[1, 1, 2, 1], strides=[1, 1, 1, 1],
                       padding="SAME",
                       expected=[2.0, 2.0, 4.0, 4.0], use_gpu=use_gpu)

  def _testMaxPoolValidPaddingUnevenStride(self, use_gpu):
    self._VerifyValues(tf.nn.max_pool, input_sizes=[1, 4, 4, 1],
                       ksize=[1, 2, 2, 1], strides=[1, 1, 2, 1],
                       padding="VALID",
                       expected=[6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                       use_gpu=use_gpu)
    self._VerifyValues(tf.nn.max_pool, input_sizes=[1, 4, 4, 1],
                       ksize=[1, 2, 2, 1], strides=[1, 2, 1, 1],
                       padding="VALID",
                       expected=[6.0, 7.0, 8.0, 14.0, 15.0, 16.0],
                       use_gpu=use_gpu)

  def _testMaxPoolSamePaddingPacket4(self, use_gpu):
    expected_output = [21.0, 22.0, 23.0, 24.0, 29.0, 30.0, 31.0, 32.0, 53.0,
                       54.0, 55.0, 56.0, 61.0, 62.0, 63.0, 64.0]
    self._VerifyValues(tf.nn.max_pool, input_sizes=[1, 4, 4, 4],
                       ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding="SAME",
                       expected=expected_output, use_gpu=use_gpu)

  def _testMaxPoolSamePaddingPacket8(self, use_gpu):
    expected_output = [145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0,
                       161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0,
                       177.0, 178.0, 179.0, 180.0, 181.0, 182.0, 183.0, 184.0,
                       185.0, 186.0, 187.0, 188.0, 189.0, 190.0, 191.0, 192.0,
                       273.0, 274.0, 275.0, 276.0, 277.0, 278.0, 279.0, 280.0,
                       289.0, 290.0, 291.0, 292.0, 293.0, 294.0, 295.0, 296.0,
                       305.0, 306.0, 307.0, 308.0, 309.0, 310.0, 311.0, 312.0,
                       313.0, 314.0, 315.0, 316.0, 317.0, 318.0, 319.0, 320.0,
                       401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0, 408.0,
                       417.0, 418.0, 419.0, 420.0, 421.0, 422.0, 423.0, 424.0,
                       433.0, 434.0, 435.0, 436.0, 437.0, 438.0, 439.0, 440.0,
                       441.0, 442.0, 443.0, 444.0, 445.0, 446.0, 447.0, 448.0,
                       465.0, 466.0, 467.0, 468.0, 469.0, 470.0, 471.0, 472.0,
                       481.0, 482.0, 483.0, 484.0, 485.0, 486.0, 487.0, 488.0,
                       497.0, 498.0, 499.0, 500.0, 501.0, 502.0, 503.0, 504.0,
                       505.0, 506.0, 507.0, 508.0, 509.0, 510.0, 511.0, 512.0]
    self._VerifyValues(tf.nn.max_pool, input_sizes=[1, 8, 8, 8],
                       ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding="SAME",
                       expected=expected_output, use_gpu=use_gpu)

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  def testMaxPooling(self):
    for use_gpu in True, False:
      self._testMaxPoolValidPadding(use_gpu)
      self._testMaxPoolSamePadding(use_gpu)
      self._testMaxPoolSamePaddingNonSquareWindow(use_gpu)
      self._testMaxPoolValidPaddingUnevenStride(use_gpu)
      self._testMaxPoolSamePaddingPacket4(use_gpu)
      self._testMaxPoolSamePaddingPacket8(use_gpu)
<<<<<<< HEAD
      self._testMaxPoolEmptyInput(use_gpu)

  # Tests for DepthwiseMaxPooling on CPU only.
  @test_util.run_deprecated_v1
=======

  # Tests for DepthwiseMaxPooling on CPU only.
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  def testDepthwiseMaxPool1x1DepthWindow1(self):
    # input is:
    # [1.0, ..., 10.0] along depth,
    #
    # We maxpool by depth in patches of 2.
<<<<<<< HEAD
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 1, 1, 10],
        ksize=[1, 1, 1, 2],
        strides=[1, 1, 1, 2],
        padding="SAME",
        expected=[2.0, 4.0, 6.0, 8.0, 10.0],
        use_gpu=False)

    for v2 in [True, False]:
      self._VerifyValues(
          gen_nn_ops.max_pool_v2,
          input_sizes=[1, 1, 1, 10],
          ksize=[1, 1, 1, 2],
          strides=[1, 1, 1, 2],
          padding="SAME",
          expected=[2.0, 4.0, 6.0, 8.0, 10.0],
          use_gpu=False,
          v2=v2)

  @test_util.run_deprecated_v1
=======
    self._VerifyValues(tf.nn.max_pool, input_sizes=[1, 1, 1, 10],
                       ksize=[1, 1, 1, 2], strides=[1, 1, 1, 2],
                       padding="SAME",
                       expected=[2.0, 4.0, 6.0, 8.0, 10.0], use_gpu=False)

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  def testDepthwiseMaxPool2x2DepthWindow3(self):
    # input is:
    #
    # a 2x2x6 cube, and we depthwise max across 3 to produce a 2x2x2
    # output.  Each node has contiguous values, so the depthwise max
    # should be multiples of 3.0.
<<<<<<< HEAD
    self._VerifyValues(
        nn_ops.max_pool,
        input_sizes=[1, 2, 2, 6],
        ksize=[1, 1, 1, 3],
        strides=[1, 1, 1, 3],
        padding="SAME",
        expected=[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0],
        use_gpu=False)

    for v2 in [True, False]:
      self._VerifyValues(
          gen_nn_ops.max_pool_v2,
          input_sizes=[1, 2, 2, 6],
          ksize=[1, 1, 1, 3],
          strides=[1, 1, 1, 3],
          padding="SAME",
          expected=[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0],
          use_gpu=False,
          v2=v2)

  @test_util.run_deprecated_v1
  def testKernelSmallerThanStrideValid(self):
    for use_gpu in [True, False]:
      self._VerifyValues(
          nn_ops.max_pool,
          input_sizes=[1, 7, 7, 1],
          ksize=[1, 2, 2, 1],
          strides=[1, 3, 3, 1],
          padding="VALID",
          expected=[9, 12, 30, 33],
          use_gpu=use_gpu)

      for v2 in [True, False]:
        self._VerifyValues(
            gen_nn_ops.max_pool_v2,
            input_sizes=[1, 7, 7, 1],
            ksize=[1, 2, 2, 1],
            strides=[1, 3, 3, 1],
            padding="VALID",
            expected=[9, 12, 30, 33],
            use_gpu=use_gpu,
            v2=v2)

      self._VerifyValues(
          nn_ops.avg_pool,
          input_sizes=[1, 7, 7, 1],
          ksize=[1, 2, 2, 1],
          strides=[1, 3, 3, 1],
          padding="VALID",
          expected=[5, 8, 26, 29],
          use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testKernelSmallerThanStrideSame(self):
    for use_gpu in [True, False]:
      for pool_func in [nn_ops.max_pool, nn_ops.avg_pool]:
        self._VerifyValues(
            pool_func,
            input_sizes=[1, 3, 3, 1],
            ksize=[1, 1, 1, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            expected=[1, 3, 7, 9],
            use_gpu=use_gpu)

        self._VerifyValues(
            pool_func,
            input_sizes=[1, 4, 4, 1],
            ksize=[1, 1, 1, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            expected=[1, 3, 9, 11],
            use_gpu=use_gpu)

      for v2 in [True, False]:
        self._VerifyValues(
            gen_nn_ops.max_pool_v2,
            input_sizes=[1, 3, 3, 1],
            ksize=[1, 1, 1, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            expected=[1, 3, 7, 9],
            use_gpu=use_gpu,
            v2=v2)

        self._VerifyValues(
            gen_nn_ops.max_pool_v2,
            input_sizes=[1, 4, 4, 1],
            ksize=[1, 1, 1, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            expected=[1, 3, 9, 11],
            use_gpu=use_gpu,
            v2=v2)

  def _testDepthwiseMaxPoolInvalidConfig(self,
                                         in_size,
                                         ksize,
                                         strides,
                                         error_msg,
                                         use_gpu=False):
    with self.cached_session(use_gpu=use_gpu):
      t = constant_op.constant(1.0, shape=in_size)
      with self.assertRaisesRegexp(errors_impl.UnimplementedError, error_msg):
        t = nn_ops.max_pool(
            t, ksize=ksize, strides=strides, padding="SAME").eval()

  @test_util.disable_xla("b/123338077")  # Passes with XLA
  def testDepthwiseMaxPoolInvalidConfigs(self):
    self._testDepthwiseMaxPoolInvalidConfig(
        [1, 2, 2, 4], [1, 2, 2, 2], [1, 1, 1, 2],
        "exactly one of pooling across depth")
    self._testDepthwiseMaxPoolInvalidConfig(
        [1, 2, 2, 4], [1, 1, 1, 2], [1, 1, 1, 1],
        "depth window to equal the depth stride")
    self._testDepthwiseMaxPoolInvalidConfig([1, 2, 2, 4], [1, 1, 1, 3],
                                            [1, 1, 1, 3], "evenly divide")
    if test.is_gpu_available():
      with self.session(use_gpu=True):
        t = variables.Variable(np.ones([1, 2, 2, 4]))
        self.evaluate(variables.global_variables_initializer())
        with self.assertRaisesOpError("for CPU devices"):
          nn_ops.max_pool(
              t, ksize=[1, 1, 1, 2], strides=[1, 1, 1, 2],
              padding="SAME").eval()

  # The following are tests that verify that the CPU and GPU implementations
  # produce the same results.
  def _CompareMaxPoolingFwd(self, input_shape, ksize, strides, padding):
    # double datatype is currently not supported for pooling ops
    # on the ROCm platform
    for dtype in [np.float32, np.float16] \
        + [np.float64] if not test.is_built_with_rocm() else []:
      tensor_input = np.random.rand(*input_shape).astype(dtype)
      with self.cached_session(use_gpu=True):
        t = constant_op.constant(tensor_input, shape=input_shape)
        out_op, _ = nn_ops.max_pool_with_argmax(t, ksize, strides, padding)
        gpu_val = self.evaluate(out_op)
      with self.cached_session(use_gpu=False):
        t = constant_op.constant(tensor_input, shape=input_shape)
        out_op = nn_ops.max_pool(t, ksize, strides, padding)
        cpu_val = self.evaluate(out_op)
      self.assertAllCloseAccordingToType(cpu_val, gpu_val)

  def _CompareMaxPoolingBk(self, input_shape, output_shape, ksize, strides,
                           padding):
    # double datatype is currently not supported for pooling ops
    # on the ROCm platform
    for dtype in [np.float32, np.float16] \
        + [np.float64] if not test.is_built_with_rocm() else []:
      # Generate numbers in a narrow range, so that there are many duplicates
      # in the input.
      tensor_input = np.random.random_integers(0, 3, input_shape).astype(dtype)
      tensor_output = np.random.rand(*output_shape).astype(dtype)
      with self.cached_session(use_gpu=True):
        t = constant_op.constant(tensor_input, shape=input_shape)
        _, argmax_op = nn_ops.max_pool_with_argmax(t, ksize, strides, padding)
        argmax = self.evaluate(argmax_op)
        grad_in = constant_op.constant(tensor_output, shape=output_shape)
        out_op = gen_nn_ops.max_pool_grad_with_argmax(t, grad_in, argmax, ksize,
                                                      strides, padding)
        gpu_val = self.evaluate(out_op)
        self.assertShapeEqual(gpu_val, out_op)
      with self.cached_session(use_gpu=False):
        t = constant_op.constant(tensor_input, shape=input_shape)
        out_op = nn_ops.max_pool(t, ksize, strides, padding)
        orig_out = self.evaluate(out_op)
        grad_in = constant_op.constant(tensor_output, shape=output_shape)
        out_op = gen_nn_ops.max_pool_grad(t, orig_out, grad_in, ksize, strides,
                                          padding)
        cpu_val = self.evaluate(out_op)
        self.assertShapeEqual(cpu_val, out_op)
      # The CPU version accumulates its gradient on fp16, so it's less
      # accurate than the GPU version that does the accumulation on fp32
      self.assertAllCloseAccordingToType(
          cpu_val, gpu_val, half_rtol=0.01, half_atol=0.01)

  def _CompareMaxPoolingGradBk(self, input_shape, output_shape, ksize, strides,
                               padding):
    # double datatype is currently not supported for pooling ops
    # on the ROCm platform
    for dtype in [np.float32, np.float16] \
        + [np.float64] if not test.is_built_with_rocm() else []:
      # Generate numbers in a narrow range, so that there are many duplicates
      # in the input.
      tensor_input = np.random.random_integers(0, 3, input_shape).astype(dtype)
      with self.cached_session(use_gpu=False):
        t = constant_op.constant(tensor_input, shape=input_shape)
        _, argmax_op = nn_ops.max_pool_with_argmax(t, ksize, strides, padding)
        argmax = self.evaluate(argmax_op)
        grad_in = constant_op.constant(tensor_input, shape=input_shape)
        out_op = gen_nn_ops.max_pool_grad_grad_with_argmax(
            t, grad_in, argmax, ksize, strides, padding)
        gpu_val = self.evaluate(out_op)
        self.assertShapeEqual(gpu_val, out_op)
      with self.cached_session(use_gpu=False):
        t = constant_op.constant(tensor_input, shape=input_shape)
        out_op = nn_ops.max_pool(t, ksize, strides, padding)
        orig_out = self.evaluate(out_op)
        grad_in = constant_op.constant(tensor_input, shape=input_shape)
        out_op = gen_nn_ops.max_pool_grad_grad(t, orig_out, grad_in, ksize,
                                               strides, padding)
        cpu_val = self.evaluate(out_op)
        self.assertShapeEqual(cpu_val, out_op)
      # The CPU version accumulates its gradient on fp16, so it's less
      # accurate than the GPU version that does the accumulation on fp32
      self.assertAllCloseAccordingToType(
          cpu_val, gpu_val, half_rtol=0.01, half_atol=0.01)

  def testMaxPoolingWithArgmax(self):
    tensor_input = [
        1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0
    ]

    Config = collections.namedtuple(
        "Config", ["use_gpu", "include_batch_in_index", "argmax"])
    configs = [
        Config(False, False, [0, 1, 3, 5, 0, 2, 6, 8]),
        Config(False, True, [0, 1, 3, 5, 9, 11, 15, 17]),
        Config(True, False, [0, 1, 3, 5, 0, 2, 6, 8]),
        Config(True, True, [0, 1, 3, 5, 9, 11, 15, 17])
    ]

    for config in configs:
      with GetDeviceScope(self, use_gpu=config.use_gpu):
        t = constant_op.constant(tensor_input, shape=[2, 3, 3, 1])
        out_op, argmax_op = nn_ops.max_pool_with_argmax(
            t,
            ksize=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            Targmax=dtypes.int64,
            padding="VALID",
            include_batch_in_index=config.include_batch_in_index)
        out, argmax = self.evaluate([out_op, argmax_op])
        self.assertShapeEqual(out, out_op)
        self.assertShapeEqual(argmax, argmax_op)
        self.assertAllClose(out.ravel(),
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertAllEqual(argmax.ravel(), config.argmax)

  def testMaxPoolingGradWithArgmax(self):
    orig_input = [
        1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0
    ]
    tensor_input = [11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0]

    Config = collections.namedtuple(
        "Config", ["use_gpu", "include_batch_in_index", "argmax"])
    configs = [
        Config(False, False, [0, 1, 3, 5, 0, 2, 6, 8]),
        Config(False, True, [0, 1, 3, 5, 9, 11, 15, 17]),
        Config(True, False, [0, 1, 3, 5, 0, 2, 6, 8]),
        Config(True, True, [0, 1, 3, 5, 9, 11, 15, 17])
    ]

    for config in configs:
      with GetDeviceScope(self, config.use_gpu):
        orig_in = constant_op.constant(orig_input, shape=[2, 3, 3, 1])
        t = constant_op.constant(tensor_input, shape=[2, 2, 2, 1])
        argmax_t = constant_op.constant(
            config.argmax, shape=[2, 2, 2, 1], dtype=dtypes.int64)
        out_op = gen_nn_ops.max_pool_grad_with_argmax(
            orig_in,
            t,
            argmax_t,
            ksize=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            include_batch_in_index=config.include_batch_in_index)
        out = self.evaluate(out_op).flatten()
        self.assertAllClose(out, [
            11.0, 12.0, 0.0, 13.0, 0.0, 14.0, 0.0, 0.0, 0.0, 21.0, 0.0, 22.0,
            0.0, 0.0, 0.0, 23.0, 0.0, 24.0
        ])

  def testMaxPoolingGradGradWithArgmax(self):
    # MaxPoolWithArgMax is implemented only on CUDA.
    if not test.is_gpu_available(cuda_only=True):
      return
    orig_input = [
        1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0
    ]
    tensor_input = [
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0
    ]

    Config = collections.namedtuple(
        "Config", ["use_gpu", "include_batch_in_index", "argmax"])
    configs = [
        Config(True, False, [0, 1, 3, 5, 0, 2, 6, 8]),
        Config(True, True, [0, 1, 3, 5, 9, 11, 15, 17])
    ]

    for config in configs:
      with GetDeviceScope(self, config.use_gpu):
        orig_in = constant_op.constant(orig_input, shape=[2, 3, 3, 1])
        t = constant_op.constant(tensor_input, shape=[2, 3, 3, 1])
        argmax_t = constant_op.constant(
            config.argmax, shape=[2, 2, 2, 1], dtype=dtypes.int64)
        out_op = gen_nn_ops.max_pool_grad_grad_with_argmax(
            orig_in,
            t,
            argmax_t,
            ksize=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            include_batch_in_index=config.include_batch_in_index)
        out = self.evaluate(out_op).flatten()
        self.assertAllClose(out,
                            [11.0, 12.0, 14.0, 16.0, 21.0, 23.0, 27.0, 29.0])

  def _ConstructAndTestGradient(self,
                                pool_func,
                                input_sizes,
                                output_sizes,
                                window_rows,
                                window_cols,
                                row_stride,
                                col_stride,
                                padding,
                                data_format,
                                use_gpu,
=======
    self._VerifyValues(tf.nn.max_pool, input_sizes=[1, 2, 2, 6],
                       ksize=[1, 1, 1, 3], strides=[1, 1, 1, 3],
                       padding="SAME",
                       expected=[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0],
                       use_gpu=False)

  def _testDepthwiseMaxPoolInvalidConfig(self, in_size, ksize, strides,
                                         error_msg, use_gpu=False):
    t = tf.constant(1.0, shape=in_size)
    with self.assertRaisesRegexp(ValueError, error_msg):
      t = tf.nn.max_pool(t, ksize=ksize, strides=strides, padding="SAME")

  def testDepthwiseMaxPoolInvalidConfigs(self):
    self._testDepthwiseMaxPoolInvalidConfig(
        [1, 2, 2, 4], [1, 2, 2, 2],
        [1, 1, 1, 2], "exactly one of pooling across depth")
    self._testDepthwiseMaxPoolInvalidConfig(
        [1, 2, 2, 4], [1, 1, 1, 2],
        [1, 1, 1, 1], "depth window to equal the depth stride")
    self._testDepthwiseMaxPoolInvalidConfig(
        [1, 2, 2, 4], [1, 1, 1, 3],
        [1, 1, 1, 3], "evenly divide")
    if tf.test.IsBuiltWithCuda():
      with self.test_session(use_gpu=True):
        t = tf.constant(1.0, shape=[1, 2, 2, 4])
        with self.assertRaisesOpError("for CPU devices"):
          tf.nn.max_pool(t, ksize=[1, 1, 1, 2], strides=[1, 1, 1, 2],
                         padding="SAME").eval()

  # The following are tests that verify that the CPU and GPU implementations
  # produce the same resuts.
  def _CompareMaxPoolingFwd(self, input_shape, ksize, strides, padding):
    tensor_input = np.random.rand(*input_shape).astype(np.float32)
    with self.test_session(use_gpu=True):
      t = tf.constant(tensor_input, shape=input_shape)
      out_op, _ = tf.nn.max_pool_with_argmax(t, ksize, strides, padding)
      gpu_val = out_op.eval()
    with self.test_session(use_gpu=False):
      t = tf.constant(tensor_input, shape=input_shape)
      out_op = tf.nn.max_pool(t, ksize, strides, padding)
      cpu_val = out_op.eval()
    self.assertAllClose(cpu_val, gpu_val, rtol=1e-5, atol=1e-5)

  def _CompareMaxPoolingBk(self, input_shape, output_shape, ksize, strides,
                           padding):
    # Generate numbers in a narrow range, so that there are many duplicates
    # in the input.
    tensor_input = np.random.random_integers(0, 3,
                                             input_shape).astype(np.float32)
    tensor_output = np.random.rand(*output_shape).astype(np.float32)
    with self.test_session(use_gpu=True):
      t = tf.constant(tensor_input, shape=input_shape)
      _, argmax_op = tf.nn.max_pool_with_argmax(t, ksize, strides, padding)
      argmax = argmax_op.eval()
      grad_in = tf.constant(tensor_output, shape=output_shape)
      out_op = gen_nn_ops._max_pool_grad_with_argmax(t, grad_in, argmax,
                                                     ksize, strides, padding)
      gpu_val = out_op.eval()
      self.assertShapeEqual(gpu_val, out_op)
    with self.test_session(use_gpu=False):
      t = tf.constant(tensor_input, shape=input_shape)
      out_op = tf.nn.max_pool(t, ksize, strides, padding)
      orig_out = out_op.eval()
      grad_in = tf.constant(tensor_output, shape=output_shape)
      out_op = gen_nn_ops._max_pool_grad(t, orig_out, grad_in, ksize,
                                         strides, padding)
      cpu_val = out_op.eval()
      self.assertShapeEqual(cpu_val, out_op)
    self.assertAllClose(cpu_val, gpu_val, rtol=1e-5, atol=1e-5)

  def testMaxPoolingWithArgmax(self):
    # MaxPoolWithArgMax is implemented only on GPU.
    if not tf.test.IsBuiltWithCuda():
      return
    tensor_input = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    with self.test_session(use_gpu=True) as sess:
      t = tf.constant(tensor_input, shape=[1, 3, 3, 1])
      out_op, argmax_op = tf.nn.max_pool_with_argmax(t,
                                                   ksize=[1, 2, 2, 1],
                                                   strides=[1, 1, 1, 1],
                                                   Targmax=tf.int64,
                                                   padding="VALID")
      out, argmax = sess.run([out_op, argmax_op])
      self.assertShapeEqual(out, out_op)
      self.assertShapeEqual(argmax, argmax_op)
      self.assertAllClose(out.ravel(), [1.0, 1.0, 1.0, 1.0])
      self.assertAllEqual(argmax.ravel(), [0, 1, 3, 5])

  def testMaxPoolingGradWithArgmax(self):
    # MaxPoolWithArgMax is implemented only on GPU.
    if not tf.test.IsBuiltWithCuda():
      return
    orig_input = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    tensor_input = [11.0, 12.0, 13.0, 14.0]
    tensor_argmax = list(np.array([0, 1, 3, 5], dtype=np.int64))
    with self.test_session(use_gpu=True) as sess:
      orig_in = tf.constant(orig_input, shape=[1, 3, 3, 1])
      t = tf.constant(tensor_input, shape=[1, 2, 2, 1])
      argmax = tf.constant(tensor_argmax, shape=[1, 2, 2, 1],
                                    dtype=tf.int64)
      out_op = gen_nn_ops._max_pool_grad_with_argmax(orig_in, t, argmax,
                                                     ksize=[1, 2, 2, 1],
                                                     strides=[1, 1, 1, 1],
                                                     padding="VALID")
      out = out_op.eval().flatten()
      self.assertAllClose(out, [11.0, 12.0, 0.0, 13.0, 0.0,
                                14.0, 0.0, 0.0, 0.0])

  def _ConstructAndTestGradient(self, pool_func, input_sizes, output_sizes,
                                window_rows, window_cols, row_stride,
                                col_stride, padding, use_gpu,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
                                x_init_value=None):
    """Verifies the gradients of the avg pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      output_sizes: Output tensor dimensions.
      window_rows: kernel size in row dim
      window_cols: kernel size in col dim
      row_stride: Row Stride.
      col_stride: Col Stride.
      padding: Padding type.
<<<<<<< HEAD
      data_format: Data format.
      use_gpu: whether we are running on GPU
      x_init_value: Values to be passed to the gradient checker.
    """
    assert input_sizes[0] == output_sizes[0]
    assert input_sizes[3] == output_sizes[3]
=======
      use_gpu: whether we are running on GPU
      x_init_value: Values to be passed to the gradient checker.
    """
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    total_size = 1
    for s in input_sizes:
      total_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x = [f * 1.0 for f in range(1, total_size + 1)]
<<<<<<< HEAD
    with self.cached_session(use_gpu=use_gpu):
      input_tensor = constant_op.constant(x, shape=input_sizes, name="input")
      if pool_func == nn_ops.avg_pool:
        func_name = "avg_pool"
        err_tolerance = 1e-4
=======
    with self.test_session(use_gpu=use_gpu):
      input_tensor = tf.constant(x, shape=input_sizes, name="input")
      if pool_func == tf.nn.avg_pool:
        func_name = "avg_pool"
        err_margin = 1e-4
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      else:
        if x_init_value is None:
          x_init_value = np.asfarray(
              np.arange(1, total_size + 1),
              dtype=np.float32).reshape(input_sizes)
        func_name = "max_pool"
<<<<<<< HEAD
        err_tolerance = 1e-3
      if data_format == "NCHW":
        ksize = [1, 1, window_rows, window_rows]
        strides = [1, 1, row_stride, col_stride]
        t = test_util.NHWCToNCHW(input_tensor)
      else:
        ksize = [1, window_rows, window_rows, 1]
        strides = [1, row_stride, col_stride, 1]
        t = input_tensor
      t = pool_func(
          t,
          ksize=ksize,
          strides=strides,
          padding=padding,
          data_format=data_format,
          name=func_name)
      if data_format == "NCHW":
        t = test_util.NCHWToNHWC(t)

      err = gradient_checker.compute_gradient_error(
          input_tensor,
          input_sizes,
          t,
          output_sizes,
          x_init_value=x_init_value,
          delta=1e-2)
    tf_logging.info("%s gradient error = %.4f" % (func_name, err))
    self.assertLess(err, err_tolerance)

  def _ConstructAndTestSecondGradient(self,
                                      pool_func,
                                      input_sizes,
                                      output_sizes,
                                      window_rows,
                                      window_cols,
                                      row_stride,
                                      col_stride,
                                      padding,
                                      data_format,
                                      use_gpu,
                                      x_init_value=None):
    """Verifies the second-order gradients of the pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      output_sizes: Output tensor dimensions.
      window_rows: kernel size in row dim
      window_cols: kernel size in col dim
      row_stride: Row Stride.
      col_stride: Col Stride.
      padding: Padding type.
      data_format: Data format.
      use_gpu: whether we are running on GPU
      x_init_value: Values to be passed to the gradient checker.
    """
    assert input_sizes[0] == output_sizes[0]
    assert input_sizes[3] == output_sizes[3]
    total_size = 1
    for s in input_sizes:
      total_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x = [f * 1.0 for f in range(1, total_size + 1)]
    with self.cached_session(use_gpu=use_gpu):
      input_tensor = constant_op.constant(x, shape=input_sizes, name="input")
      if pool_func == nn_ops.avg_pool:
        func_name = "avg_pool"
        err_tolerance = 1e-3
      else:
        if x_init_value is None:
          x_init_value = np.asfarray(
              np.arange(1, total_size + 1),
              dtype=np.float32).reshape(input_sizes)
        func_name = "max_pool"
        err_tolerance = 1e-2
      if data_format == "NCHW":
        ksize = [1, 1, window_rows, window_rows]
        strides = [1, 1, row_stride, col_stride]
        t = test_util.NHWCToNCHW(input_tensor)
      else:
        ksize = [1, window_rows, window_rows, 1]
        strides = [1, row_stride, col_stride, 1]
        t = input_tensor
      t = pool_func(
          t,
          ksize=ksize,
          strides=strides,
          padding=padding,
          data_format=data_format,
          name=func_name)
      if data_format == "NCHW":
        t = test_util.NHWCToNCHW(t)

      t_g = gradients_impl.gradients(t**2, input_tensor)[0]
      err = gradient_checker.compute_gradient_error(
          input_tensor,
          input_sizes,
          t_g,
          input_sizes,
          x_init_value=x_init_value,
          delta=1e-2)
    tf_logging.info("%s second-order gradient error = %.4f" % (func_name, err))
    self.assertLess(err, err_tolerance)

  def _testMaxPoolGradValidPadding1_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 3, 3, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=1,
          window_cols=1,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding2_1_6(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 6, 6, 3],
          output_sizes=[2, 5, 5, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding2_1_7(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 7, 7, 3],
          output_sizes=[2, 6, 6, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding1_2(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 3, 3, 1],
          output_sizes=[1, 2, 2, 1],
          window_rows=1,
          window_cols=1,
          row_stride=2,
          col_stride=2,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding2_2(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 2, 3],
          output_sizes=[2, 1, 1, 3],
          window_rows=2,
          window_cols=2,
          row_stride=2,
          col_stride=2,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding1_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 2, 4, 3],
          window_rows=1,
          window_cols=1,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding1_2(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 1, 2, 3],
          window_rows=1,
          window_cols=1,
          row_stride=2,
          col_stride=2,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding2_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 2, 4, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding2_2(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 1, 2, 3],
          window_rows=2,
          window_cols=2,
          row_stride=2,
          col_stride=2,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding3_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 7, 7, 1],
          window_rows=3,
          window_cols=3,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testMaxPoolGrad(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self._testMaxPoolGradValidPadding1_1(data_format, use_gpu)
      self._testMaxPoolGradValidPadding1_2(data_format, use_gpu)
      self._testMaxPoolGradValidPadding2_1_6(data_format, use_gpu)
      self._testMaxPoolGradValidPadding2_1_7(data_format, use_gpu)
      self._testMaxPoolGradValidPadding2_2(data_format, use_gpu)
      self._testMaxPoolGradSamePadding1_1(data_format, use_gpu)
      self._testMaxPoolGradSamePadding1_2(data_format, use_gpu)
      self._testMaxPoolGradSamePadding2_1(data_format, use_gpu)
      self._testMaxPoolGradSamePadding2_2(data_format, use_gpu)
      self._testMaxPoolGradSamePadding3_1(data_format, use_gpu)

  def _MaxPoolGrad(self, orig_input, orig_output, grad, window_rows,
                   window_cols, row_stride, col_stride, padding, v2):
=======
        err_margin = 1e-3
      t = pool_func(input_tensor, ksize=[1, window_rows, window_rows, 1],
                    strides=[1, row_stride, col_stride, 1],
                    padding=padding, name=func_name)
      err = gc.ComputeGradientError(
          input_tensor, input_sizes, t, output_sizes,
          x_init_value=x_init_value, delta=1e-2)
    print "%s gradient error = " % func_name, err
    self.assertLess(err, err_margin)

  def _testMaxPoolGradValidPadding1_1(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.max_pool, input_sizes=[1, 3, 3, 1],
        output_sizes=[1, 3, 3, 1], window_rows=1, window_cols=1, row_stride=1,
        col_stride=1, padding="VALID", use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding2_1_6(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.max_pool, input_sizes=[2, 6, 6, 3],
        output_sizes=[2, 5, 5, 3], window_rows=2, window_cols=2, row_stride=1,
        col_stride=1, padding="VALID", use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding2_1_7(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.max_pool, input_sizes=[2, 7, 7, 3],
        output_sizes=[2, 6, 6, 3], window_rows=2, window_cols=2, row_stride=1,
        col_stride=1, padding="VALID", use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding2_2(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.max_pool, input_sizes=[2, 2, 2, 3],
        output_sizes=[2, 1, 1, 3], window_rows=2, window_cols=2, row_stride=2,
        col_stride=2, padding="VALID", use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding1_1(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.max_pool, input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 2, 4, 3], window_rows=1, window_cols=1, row_stride=1,
        col_stride=1, padding="SAME", use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding2_1(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.max_pool, input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 2, 4, 3], window_rows=2, window_cols=2, row_stride=1,
        col_stride=1, padding="SAME", use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding2_2(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.max_pool, input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 1, 2, 3], window_rows=2, window_cols=2, row_stride=2,
        col_stride=2, padding="SAME", use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding3_1(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.max_pool, input_sizes=[1, 7, 7, 1],
        output_sizes=[1, 7, 7, 1], window_rows=3, window_cols=3, row_stride=1,
        col_stride=1, padding="SAME", use_gpu=use_gpu)

  def testMaxPoolGrad(self):
    for use_gpu in True, False:
      self._testMaxPoolGradValidPadding1_1(use_gpu=use_gpu)
      self._testMaxPoolGradValidPadding2_1_6(use_gpu=use_gpu)
      self._testMaxPoolGradValidPadding2_1_7(use_gpu=use_gpu)
      self._testMaxPoolGradValidPadding2_2(use_gpu=use_gpu)
      self._testMaxPoolGradSamePadding1_1(use_gpu=use_gpu)
      self._testMaxPoolGradSamePadding2_1(use_gpu=use_gpu)
      self._testMaxPoolGradSamePadding2_2(use_gpu=use_gpu)
      self._testMaxPoolGradSamePadding3_1(use_gpu=use_gpu)

  def _MaxPoolGrad(self, orig_input, orig_output, grad, window_rows,
                   window_cols, row_stride, col_stride, padding):
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    """Max Pooling Gradient.

    Args:
      orig_input: A float Tensor. The original input tensor.
      orig_output: A float Tensor. The original output tensor.
      grad: A float Tensor.
        The 4D (batch x rows x cols x depth) output backprop.
      window_rows: integer. Kernel size along rows dimension.
      window_cols: integer. Kernel size along cols dimension.
      row_stride: integer. Stride along rows dimension
      col_stride: integer. Stride along cols dimension
      padding: PoolingOpDef.Padding.  Padding type.

    Returns:
      A Tensor.
    """
<<<<<<< HEAD
    pool_func = gen_nn_ops.max_pool_grad_v2 if v2 else gen_nn_ops.max_pool_grad
    return pool_func(orig_input, orig_output, grad,
                     [1, window_rows, window_cols, 1],
                     [1, row_stride, col_stride, 1], padding)
=======
    return gen_nn_ops._max_pool_grad(
        orig_input, orig_output, grad,
        [1, window_rows, window_cols, 1], [1, row_stride, col_stride, 1],
        padding)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def _testMaxPoolGradDirect(self, input_data, output_backprop,
                             expected_input_backprop, input_sizes, output_sizes,
                             window_rows, window_cols, row_stride, col_stride,
<<<<<<< HEAD
                             padding, use_gpu, v2):
    pool_func = gen_nn_ops.max_pool_v2 if v2 else nn_ops.max_pool
    with self.cached_session(use_gpu=use_gpu):
      input_tensor = variables.Variable(
          np.array(input_data, dtype=np.float32).reshape(input_sizes))
      self.evaluate(variables.global_variables_initializer())
      output_tensor = pool_func(input_tensor, [1, window_rows, window_cols, 1],
                                [1, row_stride, col_stride, 1], padding)
      output_backprop_tensor = constant_op.constant(
          output_backprop, shape=output_sizes)

      input_backprop_tensor = self._MaxPoolGrad(
          input_tensor, output_tensor, output_backprop_tensor, window_rows,
          window_cols, row_stride, col_stride, padding, v2)

      actual_input_backprop = self.evaluate(input_backprop_tensor)
=======
                             padding, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      input_tensor = tf.constant(input_data, shape=input_sizes)
      output_tensor = tf.nn.max_pool(
          input_tensor, [1, window_rows, window_cols, 1],
          [1, row_stride, col_stride, 1], padding)
      output_backprop_tensor = tf.constant(output_backprop,
                                                    shape=output_sizes)

      input_backprop_tensor = self._MaxPoolGrad(
          input_tensor, output_tensor, output_backprop_tensor,
          window_rows, window_cols, row_stride, col_stride, padding)

      actual_input_backprop = input_backprop_tensor.eval()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      self.assertShapeEqual(actual_input_backprop, input_backprop_tensor)
      actual_input_backprop = actual_input_backprop.flatten()
      actual_input_backprop = self._GetNdArray(actual_input_backprop)

<<<<<<< HEAD
      actual_output = self.evaluate(output_tensor).flatten()
      actual_output = self._GetNdArray(actual_output)

      self.assertAllClose(
          expected_input_backprop, actual_input_backprop, rtol=1e-6, atol=1e-6)

  def _testMaxPoolGradDirect1_1(self):
    input_data = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0]
    expected_input_backprop = [
        11.0, 12.0, 13.0, 0.0, 15.0, 16.0, 17.0, 0.0, 19.0, 20.0, 21.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ]

    for use_gpu in True, False:
      for v2 in [True, False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 3, 1],
            window_rows=2,
            window_cols=2,
            row_stride=1,
            col_stride=1,
            padding="VALID",
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradDirect1_2(self):
    input_data = [
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0]
    expected_input_backprop = [
        11.0, 0.0, 25.0, 0.0, 0.0, 31.0, 0.0, 17.0, 19.0, 0.0, 41.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ]

    for use_gpu in True, False:
      for v2 in [True, False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 3, 1],
            window_rows=2,
            window_cols=2,
            row_stride=1,
            col_stride=1,
            padding="VALID",
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradDirect1_3(self):
    input_data = [
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
    ]
    output_backprop = [
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
        23.0, 24.0, 25.0, 26.0
    ]
    expected_input_backprop = [
        54,
        0.0,
        62,
        0.0,
        0.0,
        60,
        0.0,
        22.0,
        47,
        0.0,
        51,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    for use_gpu in True, False:
      for v2 in [True, False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 4, 4, 1],
            window_rows=3,
            window_cols=3,
            row_stride=1,
            col_stride=1,
            padding="SAME",
            use_gpu=use_gpu,
            v2=v2)

  @test_util.no_xla_auto_jit("b/123923733")  # NaNs handled differently
  def _testMaxPoolGradDirectWithNans2_1(self):
    input_data = [float("nan")] * 16
    output_backprop = [11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0]
    # Test the CPU implementation, which propagates diffs in case of NaN
    expected_input_backprop_tf_cpu = [
        11.0, 12.0, 13.0, 0.0, 15.0, 16.0, 17.0, 0.0, 19.0, 20.0, 21.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ]
    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_tf_cpu,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=False,
          v2=v2)

    if not test.is_gpu_available():
      return

    # The functionality associated with TF_ENABLE_NANPROP is currently
    # not supported on the ROCm platform, so skip this part of the test
    # NANs in input lead to non-deterministic results, and hence skipping
    # the remaining tests altogeher on the ROCm platform
    if test.is_built_with_rocm():
      return

    # Test the GPU implementation that uses cudnn for now.
    saved_nanprop = os.environ.get("TF_ENABLE_MAXPOOL_NANPROP")
    # Do not propagate the diff in cases of NaNs
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "0"
    expected_input_backprop_cudnn = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ]

    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_cudnn,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=True,
          v2=v2)

    # Propagate the diff in cases of NaNs
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "1"
    expected_input_backprop_cudnn = expected_input_backprop_tf_cpu

    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_cudnn,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=True,
          v2=v2)

    if saved_nanprop:
      os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = saved_nanprop
    else:
      del os.environ["TF_ENABLE_MAXPOOL_NANPROP"]

  @test_util.no_xla_auto_jit("b/123923733")  # NaNs handled differently
  def _testMaxPoolGradDirectWithNans2_2(self):
    input_data = [float("nan")] * 16
    output_backprop = [
        float("nan"), 12.0, 13.0, 15.0,
        float("nan"), 17.0, 19.0, 20.0,
        float("nan")
    ]
    # Test the CPU implementation, which propagates diffs in case of NaN
    expected_input_backprop_tf_cpu = [
        float("nan"), 12.0, 13.0, 0.0, 15.0,
        float("nan"), 17.0, 0.0, 19.0, 20.0,
        float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_tf_cpu,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=False,
          v2=v2)

    if not test.is_gpu_available():
      return

    # The functionality associated with TF_ENABLE_NANPROP is currently
    # not supported on the ROCm platform, so skip this part of the test
    # NANs in input lead to non-deterministic results, and hence skipping
    # the remaining tests altogeher on the ROCm platform
    if test.is_built_with_rocm():
      return

    # Test the GPU implementation that uses cudnn for now.
    saved_nanprop = os.environ.get("TF_ENABLE_MAXPOOL_NANPROP")
    # Do not propagate the diff in cases of NaNs
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "0"
    expected_input_backprop_cudnn = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ]

    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_cudnn,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=True,
          v2=v2)

    # Propagate the diff in cases of NaNs
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "1"
    expected_input_backprop_cudnn = expected_input_backprop_tf_cpu

    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_cudnn,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=True,
          v2=v2)

    if saved_nanprop:
      os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = saved_nanprop
    else:
      del os.environ["TF_ENABLE_MAXPOOL_NANPROP"]

  @test_util.run_deprecated_v1
=======
      actual_output = output_tensor.eval().flatten()
      actual_output = self._GetNdArray(actual_output)

      self.assertAllClose(expected_input_backprop, actual_input_backprop,
                          rtol=1e-6, atol=1e-6)

  def _testMaxPoolGradDirect1_1(self):
    input_data = [
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0]
    output_backprop = [
        11.0, 12.0, 13.0,
        15.0, 16.0, 17.0,
        19.0, 20.0, 21.0]
    expected_input_backprop = [
        11.0, 12.0, 13.0, 0.0,
        15.0, 16.0, 17.0, 0.0,
        19.0, 20.0, 21.0, 0.0,
        0.0, 0.0, 0.0, 0.0]

    for use_gpu in True, False:
      self._testMaxPoolGradDirect(
          input_data, output_backprop, expected_input_backprop,
          input_sizes=[1, 4, 4, 1], output_sizes=[1, 3, 3, 1],
          window_rows=2, window_cols=2, row_stride=1, col_stride=1,
          padding="VALID", use_gpu=use_gpu)

  def _testMaxPoolGradDirect1_2(self):
    input_data = [
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0]
    output_backprop = [
        11.0, 12.0, 13.0,
        15.0, 16.0, 17.0,
        19.0, 20.0, 21.0]
    expected_input_backprop = [
        11.0, 0.0, 25.0, 0.0,
        0.0, 31.0, 0.0, 17.0,
        19.0, 0.0, 41.0, 0.0,
        0.0, 0.0, 0.0, 0.0]

    for use_gpu in True, False:
      self._testMaxPoolGradDirect(
          input_data, output_backprop, expected_input_backprop,
          input_sizes=[1, 4, 4, 1], output_sizes=[1, 3, 3, 1],
          window_rows=2, window_cols=2, row_stride=1, col_stride=1,
          padding="VALID", use_gpu=use_gpu)

  def _testMaxPoolGradDirect1_3(self):
    input_data = [
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0,]
    output_backprop = [
        11.0, 12.0, 13.0, 14.0,
        15.0, 16.0, 17.0, 18.0,
        19.0, 20.0, 21.0, 22.0,
        23.0, 24.0, 25.0, 26.0]
    expected_input_backprop = [
        54, 0.0, 62, 0.0,
        0.0, 60, 0.0, 22.0,
        47, 0.0, 51, 0.0,
        0.0, 0.0, 0.0, 0.0,]

    for use_gpu in True, False:
      self._testMaxPoolGradDirect(
          input_data, output_backprop, expected_input_backprop,
          input_sizes=[1, 4, 4, 1], output_sizes=[1, 4, 4, 1],
          window_rows=3, window_cols=3, row_stride=1, col_stride=1,
          padding="SAME", use_gpu=use_gpu)

  def _testMaxPoolGradDirectWithNans2_1(self):
    input_data = [float("nan")] * 16
    output_backprop = [
        11.0, 12.0, 13.0,
        15.0, 16.0, 17.0,
        19.0, 20.0, 21.0]
    # Test the CPU implementation, which propagates diffs in case of NaN
    expected_input_backprop_tf_cpu = [
        11.0, 12.0, 13.0, 0.0,
        15.0, 16.0, 17.0, 0.0,
        19.0, 20.0, 21.0, 0.0,
        0.0, 0.0, 0.0, 0.0]
    self._testMaxPoolGradDirect(
        input_data, output_backprop, expected_input_backprop_tf_cpu,
        input_sizes=[1, 4, 4, 1], output_sizes=[1, 3, 3, 1],
        window_rows=2, window_cols=2, row_stride=1, col_stride=1,
        padding="VALID", use_gpu=False)

    if not tf.test.IsBuiltWithCuda():
      return

    # Test the GPU implementation that uses cudnn for now.
    # It does not propagate the diff in cases of NaNs
    expected_input_backprop_cudnn = [
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0]
    self._testMaxPoolGradDirect(
        input_data, output_backprop, expected_input_backprop_cudnn,
        input_sizes=[1, 4, 4, 1], output_sizes=[1, 3, 3, 1],
        window_rows=2, window_cols=2, row_stride=1, col_stride=1,
        padding="VALID", use_gpu=True)

  def _testMaxPoolGradDirectWithNans2_2(self):
    input_data = [float("nan")] * 16
    output_backprop = [
        float("nan"), 12.0, 13.0,
        15.0, float("nan"), 17.0,
        19.0, 20.0, float("nan")]
    # Test the CPU implementation, which propagates diffs in case of NaN
    expected_input_backprop_tf_cpu = [
        float("nan"), 12.0, 13.0, 0.0,
        15.0, float("nan"), 17.0, 0.0,
        19.0, 20.0, float("nan"), 0.0,
        0.0, 0.0, 0.0, 0.0]
    self._testMaxPoolGradDirect(
        input_data, output_backprop, expected_input_backprop_tf_cpu,
        input_sizes=[1, 4, 4, 1], output_sizes=[1, 3, 3, 1],
        window_rows=2, window_cols=2, row_stride=1, col_stride=1,
        padding="VALID", use_gpu=False)

    if not tf.test.IsBuiltWithCuda():
      return

    # Test the GPU implementation that uses cudnn for now.
    # It does not propagate the diff in cases of NaNs
    expected_input_backprop_cudnn = [
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0]
    self._testMaxPoolGradDirect(
        input_data, output_backprop, expected_input_backprop_cudnn,
        input_sizes=[1, 4, 4, 1], output_sizes=[1, 3, 3, 1],
        window_rows=2, window_cols=2, row_stride=1, col_stride=1,
        padding="VALID", use_gpu=True)

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  def testMaxPoolGradDirect(self):
    self._testMaxPoolGradDirect1_1()
    self._testMaxPoolGradDirect1_2()
    self._testMaxPoolGradDirect1_3()
    self._testMaxPoolGradDirectWithNans2_1()
    self._testMaxPoolGradDirectWithNans2_2()

<<<<<<< HEAD
  def _testMaxPoolGradGradValidPadding1_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[1, 3, 3, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=1,
          window_cols=1,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradValidPadding2_1_6(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 6, 6, 3],
          output_sizes=[2, 5, 5, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradValidPadding2_1_7(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 7, 7, 3],
          output_sizes=[2, 6, 6, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradValidPadding2_2(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 2, 2, 3],
          output_sizes=[2, 1, 1, 3],
          window_rows=2,
          window_cols=2,
          row_stride=2,
          col_stride=2,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradSamePadding1_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 2, 4, 3],
          window_rows=1,
          window_cols=1,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradSamePadding2_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 2, 4, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradSamePadding2_2(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 1, 2, 3],
          window_rows=2,
          window_cols=2,
          row_stride=2,
          col_stride=2,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradSamePadding3_1(self, data_format, use_gpu):
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 7, 7, 1],
          window_rows=3,
          window_cols=3,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testMaxPoolGradGrad(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self._testMaxPoolGradGradValidPadding1_1(data_format, use_gpu)
      self._testMaxPoolGradGradValidPadding2_1_6(data_format, use_gpu)
      self._testMaxPoolGradGradValidPadding2_1_7(data_format, use_gpu)
      self._testMaxPoolGradGradValidPadding2_2(data_format, use_gpu)
      self._testMaxPoolGradGradSamePadding1_1(data_format, use_gpu)
      self._testMaxPoolGradGradSamePadding2_1(data_format, use_gpu)
      self._testMaxPoolGradGradSamePadding2_2(data_format, use_gpu)
      self._testMaxPoolGradGradSamePadding3_1(data_format, use_gpu)

  def _MaxPoolGradGrad(self, orig_input, orig_output, grad, window_rows,
                       window_cols, row_stride, col_stride, padding):
    """Max Pooling Second-Order Gradient.

    Args:
      orig_input: A float Tensor. The original input tensor.
      orig_output: A float Tensor. The original output tensor.
      grad: A float Tensor.
        The 4D (batch x out_rows x out_cols x depth) output backprop.
      window_rows: integer. Kernel size along rows dimension.
      window_cols: integer. Kernel size along cols dimension.
      row_stride: integer. Stride along rows dimension
      col_stride: integer. Stride along cols dimension
      padding: PoolingOpDef.Padding.  Padding type.

    Returns:
      A Tensor.
    """
    return gen_nn_ops.max_pool_grad_grad(
        orig_input, orig_output, grad, [1, window_rows, window_cols, 1],
        [1, row_stride, col_stride, 1], padding)

  @test_util.run_deprecated_v1
  def testAvgPoolGrad(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self._testAvgPoolGradValidPadding1_1(data_format, use_gpu)
      self._testAvgPoolGradValidPadding1_2(data_format, use_gpu)
      self._testAvgPoolGradValidPadding2_1(data_format, use_gpu)
      self._testAvgPoolGradValidPadding2_2(data_format, use_gpu)
      self._testAvgPoolGradSamePadding1_1(data_format, use_gpu)
      self._testAvgPoolGradSamePadding1_2(data_format, use_gpu)
      self._testAvgPoolGradSamePadding2_1(data_format, use_gpu)
      self._testAvgPoolGradSamePadding2_2(data_format, use_gpu)
      self._testAvgPoolGradSamePadding3_1(data_format, use_gpu)

  def _testAvgPoolGradValidPadding1_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 3, 3, 3],
        output_sizes=[2, 3, 3, 3],
        window_rows=1,
        window_cols=1,
        row_stride=1,
        col_stride=1,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding1_2(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 3, 3, 3],
        output_sizes=[2, 2, 2, 3],
        window_rows=1,
        window_cols=1,
        row_stride=2,
        col_stride=2,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding2_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 3, 3, 3],
        output_sizes=[2, 2, 2, 3],
        window_rows=2,
        window_cols=2,
        row_stride=1,
        col_stride=1,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding2_2(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 2, 3],
        output_sizes=[2, 1, 1, 3],
        window_rows=2,
        window_cols=2,
        row_stride=2,
        col_stride=2,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding1_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 2, 4, 3],
        window_rows=1,
        window_cols=1,
        row_stride=1,
        col_stride=1,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding1_2(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 1, 2, 3],
        window_rows=1,
        window_cols=1,
        row_stride=2,
        col_stride=2,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding2_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 2, 4, 3],
        window_rows=2,
        window_cols=2,
        row_stride=1,
        col_stride=1,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding2_2(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 1, 2, 3],
        window_rows=2,
        window_cols=2,
        row_stride=2,
        col_stride=2,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding3_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[1, 7, 7, 1],
        output_sizes=[1, 7, 7, 1],
        window_rows=3,
        window_cols=3,
        row_stride=1,
        col_stride=1,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testShapeFunctionEdgeCases(self):
    # All shapes unknown.
    for pool_func in [nn_ops.max_pool, nn_ops.avg_pool]:
      p = pool_func(
          array_ops.placeholder(dtypes.float32),
          ksize=[1, 1, 1, 1],
          strides=[1, 1, 1, 1],
          padding="SAME")
      self.assertEqual([None, None, None, None], p.get_shape().as_list())
    p, am = nn_ops.max_pool_with_argmax(
        array_ops.placeholder(dtypes.float32),
        ksize=[1, 1, 1, 1],
        strides=[1, 1, 1, 1],
=======
  def testAvgPoolGrad(self):
    for use_gpu in False, True:
      self._testAvgPoolGradValidPadding1_1(use_gpu)
      self._testAvgPoolGradValidPadding2_1(use_gpu)
      self._testAvgPoolGradValidPadding2_2(use_gpu)
      self._testAvgPoolGradSamePadding1_1(use_gpu)
      self._testAvgPoolGradSamePadding2_1(use_gpu)
      self._testAvgPoolGradSamePadding2_2(use_gpu)
      self._testAvgPoolGradSamePadding3_1(use_gpu)

  def _testAvgPoolGradValidPadding1_1(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.avg_pool, input_sizes=[2, 3, 3, 3],
        output_sizes=[2, 3, 3, 3], window_rows=1, window_cols=1, row_stride=1,
        col_stride=1, padding="VALID", use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding2_1(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.avg_pool, input_sizes=[2, 3, 3, 3],
        output_sizes=[2, 2, 2, 3], window_rows=2, window_cols=2, row_stride=1,
        col_stride=1, padding="VALID", use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding2_2(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.avg_pool, input_sizes=[2, 2, 2, 3],
        output_sizes=[2, 1, 1, 3], window_rows=2, window_cols=2, row_stride=2,
        col_stride=2, padding="VALID", use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding1_1(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.avg_pool, input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 2, 4, 3], window_rows=1, window_cols=1, row_stride=1,
        col_stride=1, padding="SAME", use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding2_1(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.avg_pool, input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 2, 4, 3], window_rows=2, window_cols=2, row_stride=1,
        col_stride=1, padding="SAME", use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding2_2(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.avg_pool, input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 1, 2, 3], window_rows=2, window_cols=2, row_stride=2,
        col_stride=2, padding="SAME", use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding3_1(self, use_gpu):
    self._ConstructAndTestGradient(
        tf.nn.avg_pool, input_sizes=[1, 7, 7, 1],
        output_sizes=[1, 7, 7, 1], window_rows=3, window_cols=3, row_stride=1,
        col_stride=1, padding="SAME", use_gpu=use_gpu)

  def testShapeFunctionEdgeCases(self):
    # All shapes unknown.
    for pool_func in [tf.nn.max_pool, tf.nn.avg_pool]:
      p = tf.nn.max_pool(tf.placeholder(tf.float32),
                         ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
                         padding="SAME")
      self.assertEqual([None, None, None, None], p.get_shape().as_list())
    p, am = tf.nn.max_pool_with_argmax(
        tf.placeholder(tf.float32),
        ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
        padding="SAME")
    self.assertEqual([None, None, None, None], p.get_shape().as_list())
    self.assertEqual([None, None, None, None], am.get_shape().as_list())

    # Incorrect input shape.
<<<<<<< HEAD
    for pool_func in [
        nn_ops.max_pool, nn_ops.avg_pool, nn_ops.max_pool_with_argmax
    ]:
      with self.assertRaises(ValueError):
        pool_func(
            array_ops.placeholder(dtypes.float32, shape=[1, 3]),
            ksize=[1, 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding="SAME")

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/123337890")  # Error messages differ
  def testOpEdgeCases(self):
    with self.session(use_gpu=test.is_gpu_available()) as sess:
      pool_funcs = [nn_ops.max_pool, nn_ops.avg_pool]
      if test.is_gpu_available():
        pool_funcs.append(nn_ops.max_pool_with_argmax)
      for pool_func in pool_funcs:
        if pool_func != nn_ops.max_pool:
          # Illegal strides.
          with self.assertRaisesRegexp(
              errors_impl.UnimplementedError,
              "Pooling is not yet supported on the batch"):
            sess.run(
                pool_func(
                    array_ops.placeholder(dtypes.float32),
                    ksize=[1, 1, 1, 1],
                    strides=[2, 1, 1, 1],
                    padding="SAME"))

        # Filter too large.
        with self.assertRaisesRegexp(ValueError, "Negative dimension size"):
          sess.run(
              pool_func(
                  array_ops.placeholder(dtypes.float32, shape=[32, 20, 20, 3]),
                  ksize=[1, 20, 21, 1],
                  strides=[1, 1, 1, 1],
                  padding="VALID"))
        with self.assertRaisesRegexp(ValueError, "Negative dimension size"):
          pool_func(
              array_ops.placeholder(dtypes.float32, shape=[32, 20, 20, 3]),
              ksize=[1, 21, 20, 1],
              strides=[1, 1, 1, 1],
              padding="VALID")


def GetMaxPoolFwdTest(input_size, filter_size, strides, padding):

  def Test(self):
    # MaxPoolWithArgMax is implemented only on CUDA.
    if not test.is_gpu_available(cuda_only=True):
      return
    self._CompareMaxPoolingFwd(input_size, filter_size, strides, padding)

=======
    for pool_func in [tf.nn.max_pool, tf.nn.avg_pool,
                      tf.nn.max_pool_with_argmax]:
      with self.assertRaises(ValueError):
        pool_func(tf.placeholder(tf.float32, shape=[1, 3]),
                  ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding="SAME")

    # Illegal strides.
    for pool_func in [tf.nn.max_pool, tf.nn.avg_pool,
                      tf.nn.max_pool_with_argmax]:
      with self.assertRaisesRegexp(ValueError, "strides in the batch"):
        pool_func(tf.placeholder(tf.float32),
                  ksize=[1, 1, 1, 1], strides=[2, 1, 1, 1], padding="SAME")
    with self.assertRaisesRegexp(ValueError, "strides in the batch and depth"):
      tf.nn.avg_pool(tf.placeholder(tf.float32),
                     ksize=[1, 1, 1, 1], strides=[1, 1, 1, 2], padding="SAME")

    # Filter larger than input.
    for pool_func in [tf.nn.max_pool, tf.nn.avg_pool,
                      tf.nn.max_pool_with_argmax]:
      with self.assertRaisesRegexp(ValueError,
                                   "filter must not be larger than the input"):
        pool_func(tf.placeholder(tf.float32,
                                        shape=[32, 20, 20, 3]),
                  ksize=[1, 20, 21, 1], strides=[1, 1, 1, 1], padding="SAME")
      with self.assertRaisesRegexp(ValueError,
                                   "filter must not be larger than the input"):
        pool_func(tf.placeholder(tf.float32,
                                        shape=[32, 20, 20, 3]),
                  ksize=[1, 21, 20, 1], strides=[1, 1, 1, 1], padding="SAME")

    # Stride larger than filter.
    for pool_func in [tf.nn.max_pool, tf.nn.avg_pool,
                      tf.nn.max_pool_with_argmax]:
      with self.assertRaisesRegexp(
          ValueError, "stride must be less than or equal to filter"):
        pool_func(tf.placeholder(tf.float32,
                                        shape=[32, 20, 20, 3]),
                  ksize=[1, 5, 3, 1], strides=[1, 5, 5, 1], padding="SAME")
      with self.assertRaisesRegexp(
          ValueError, "stride must be less than or equal to filter"):
        pool_func(tf.placeholder(tf.float32,
                                        shape=[32, 20, 20, 3]),
                  ksize=[1, 3, 5, 1], strides=[1, 5, 5, 1], padding="SAME")


def GetMaxPoolFwdTest(input_size, filter_size, strides, padding):
  def Test(self):
    # MaxPoolWithArgMax is implemented only on GPU.
    if not tf.test.IsBuiltWithCuda():
      return
    self._CompareMaxPoolingFwd(input_size, filter_size, strides, padding)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  return Test


def GetMaxPoolGradTest(input_size, filter_size, output_size, strides, padding):
<<<<<<< HEAD

  def Test(self):
    # MaxPoolWithArgMax is implemented only on CUDA.
    if not test.is_gpu_available(cuda_only=True):
      return
    self._CompareMaxPoolingBk(input_size, output_size, filter_size, strides,
                              padding)

  return Test


def GetMaxPoolGradGradTest(input_size, filter_size, output_size, strides,
                           padding):

  def Test(self):
    # MaxPoolWithArgMax is implemented only on CUDA.
    if not test.is_gpu_available(cuda_only=True):
      return
    self._CompareMaxPoolingGradBk(input_size, output_size, filter_size, strides,
                                  padding)

=======
  def Test(self):
    # MaxPoolWithArgMax is implemented only on GPU.
    if not tf.test.IsBuiltWithCuda():
      return
    self._CompareMaxPoolingBk(input_size, output_size,
                              filter_size, strides, padding)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  return Test


if __name__ == "__main__":
  for (name_, input_size_, filter_size_, output_size_, stride_,
<<<<<<< HEAD
       padding_) in GetShrunkInceptionMaxPoolShapes():
    setattr(PoolingTest, "testMaxPoolFwd_" + name_,
            GetMaxPoolFwdTest(input_size_, filter_size_, stride_, padding_))
    setattr(PoolingTest, "testMaxPoolGrad_" + name_,
            GetMaxPoolGradTest(input_size_, filter_size_, output_size_, stride_,
                               padding_))
    setattr(PoolingTest, "testMaxPoolGradGrad_" + name_,
            GetMaxPoolGradGradTest(input_size_, filter_size_, output_size_,
                                   stride_, padding_))
  test.main()
=======
       padding_) in GetInceptionMaxPoolShapes():
    setattr(PoolingTest, "testMaxPoolFwd_" + name_,
            GetMaxPoolFwdTest(input_size_, filter_size_, stride_, padding_))
    setattr(PoolingTest, "testMaxPoolGrad_" + name_,
            GetMaxPoolGradTest(input_size_, filter_size_, output_size_,
                               stride_, padding_))
  tf.test.main()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
