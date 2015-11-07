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
"""Tests for PrecisionOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class InTopKTest(test.TestCase):

  def _validateInTopK(self, predictions, target, k, expected):
    np_ans = np.array(expected)
    with self.cached_session(use_gpu=True):
      precision = nn_ops.in_top_k(predictions, target, k)
      out = self.evaluate(precision)
=======
"""Tests for PrecisionOp."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class InTopKTest(tf.test.TestCase):

  def _validateInTopK(self, predictions, target, k, expected):
    np_ans = np.array(expected)
    with self.test_session():
      precision = tf.nn.in_top_k(predictions, target, k)
      out = precision.eval()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      self.assertAllClose(np_ans, out)
      self.assertShapeEqual(np_ans, precision)

  def testInTop1(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
<<<<<<< HEAD
    target = [3, 2]
=======
    target = [3, 1]
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    self._validateInTopK(predictions, target, 1, [True, False])

  def testInTop2(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
<<<<<<< HEAD
    target = [2, 2]
=======
    target = [0, 2]
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    self._validateInTopK(predictions, target, 2, [False, True])

  def testInTop2Tie(self):
    # Class 2 and 3 tie for 2nd, so both are considered in top 2.
    predictions = [[0.1, 0.3, 0.2, 0.2], [0.1, 0.3, 0.2, 0.2]]
    target = [2, 3]
    self._validateInTopK(predictions, target, 2, [True, True])

<<<<<<< HEAD
  def testInTop2_int64Target(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = np.asarray([0, 2]).astype(np.int64)
    self._validateInTopK(predictions, target, 2, [False, True])

  def testInTopNan(self):
    predictions = [[0.1, float("nan"), 0.2, 0.4], [0.1, 0.2, 0.3, float("inf")]]
    target = [1, 3]
    self._validateInTopK(predictions, target, 2, [False, False])

  def testBadTarget(self):
    predictions = [[0.1, 0.3, 0.2, 0.2], [0.1, 0.3, 0.2, 0.2]]
    target = [2, 4]  # must return False for invalid target
    self._validateInTopK(predictions, target, 2, [True, False])

  def testTensorK(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [0, 2]
    k = constant_op.constant(3)
    np_ans = np.array([False, True])
    with self.cached_session():
      precision = nn_ops.in_top_k(predictions, target, k)
      out = self.evaluate(precision)
      self.assertAllClose(np_ans, out)
      self.assertShapeEqual(np_ans, precision)


if __name__ == "__main__":
  test.main()
=======

if __name__ == "__main__":
  tf.test.main()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
