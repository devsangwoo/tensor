# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Basic tests for TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ReshapeTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 24, 24, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      outputs = []
      # Here we test two types of reshapes, one changes the batch dimension and
      # the other does not. Note that we're not able to test reshaping to
      # scalar, since TRT requires input tensor to be of rank at least 2, so a
      # reshape with scalar input will be filtered out of the segment before
      # conversion.
      with g.device("/GPU:0"):
        # These reshapes happen at batch dimension, thus should fail.
        for shape in [[2, 50, 24, 24, 2], [-1, 50, 24, 24, 2],
                      [2, 50, -1, 24, 2]]:
          r = array_ops.reshape(inp, shape)
          r = array_ops.reshape(r, [-1, 24, 24, 2])
          outputs.append(self.trt_incompatible_op(r))
        # Add another block with many reshapes that don't change the batch
        # dimension.
        r = array_ops.reshape(inp, [-1, 24 * 24, 2], name="reshape-0")
        r = array_ops.reshape(r, [100, 24, -1], name="reshape-1")
        r = array_ops.reshape(r, [100, 24 * 2, 24], name="reshape-2")
        r = array_ops.reshape(r, [-1, 24, 24 * 2], name="reshape-3")
        r = array_ops.reshape(r, [-1, 6, 4, 24, 2], name="reshape-4")
        r = array_ops.reshape(r, [-1, 6, 4, 6, 4, 2, 1], name="reshape-5")
        r = array_ops.reshape(r, [-1, 24, 24, 2], name="reshape-6")
        outputs.append(self.trt_incompatible_op(r))
      math_ops.add_n(outputs, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[tuple(input_dims)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "my_trt_op_3": ["reshape-%d" % i for i in range(7)] +
                       ["reshape-%d/shape" % i for i in range(7)]
    }

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    return (not trt_test.IsQuantizationMode(run_params.precision_mode) and
            not run_params.dynamic_engine)


class TransposeTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Create a graph containing single segment."""
    dtype = dtypes.float32
    input_name = "input"
    input_dims = [100, 24, 24, 2]
    output_name = "output"
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtype, shape=[None] + input_dims[1:], name=input_name)
      with g.device("/GPU:0"):
        t = array_ops.transpose(inp, [0, 3, 1, 2], name="transpose-1")
        t = array_ops.transpose(t, [0, 2, 3, 1], name="transposeback-1")
        incompatible = self.trt_incompatible_op(t)
        t = array_ops.transpose(incompatible, [2, 1, 0, 3], name="transpose-2")
        t = array_ops.transpose(t, [0, 2, 3, 1], name="transpose-3")
      array_ops.identity(t, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(24, 100, 2, 24)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return {
        "my_trt_op_0": [
            "transpose-1", "transpose-1/perm", "transposeback-1",
            "transposeback-1/perm"
        ]
    }

  def ShouldRunTest(self, run_params):
    """Whether to run the test."""
    return (not trt_test.IsQuantizationMode(run_params.precision_mode) and
            not run_params.dynamic_engine)


if __name__ == "__main__":
  test.main()
