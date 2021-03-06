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
"""Tests for tensorflow.ops.numerics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import numerics
from tensorflow.python.platform import test


class VerifyTensorAllFiniteTest(test.TestCase):
=======
"""Tests for tensorflow.ops.numerics."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops


class VerifyTensorAllFiniteTest(tf.test.TestCase):
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def testVerifyTensorAllFiniteSucceeds(self):
    x_shape = [5, 4]
    x = np.random.random_sample(x_shape).astype(np.float32)
<<<<<<< HEAD
    with test_util.use_gpu():
      t = constant_op.constant(x, shape=x_shape, dtype=dtypes.float32)
      t_verified = numerics.verify_tensor_all_finite(t,
                                                     "Input is not a number.")
      self.assertAllClose(x, self.evaluate(t_verified))
=======
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        t = tf.constant(x, shape=x_shape, dtype=tf.float32)
        t_verified = tf.verify_tensor_all_finite(t, "Input is not a number.")
        self.assertAllClose(x, t_verified.eval())
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def testVerifyTensorAllFiniteFails(self):
    x_shape = [5, 4]
    x = np.random.random_sample(x_shape).astype(np.float32)
    my_msg = "Input is not a number."

    # Test NaN.
    x[0] = np.nan
<<<<<<< HEAD
    with test_util.use_gpu():
      with self.assertRaisesOpError(my_msg):
        t = constant_op.constant(x, shape=x_shape, dtype=dtypes.float32)
        t_verified = numerics.verify_tensor_all_finite(t, my_msg)
        self.evaluate(t_verified)

    # Test Inf.
    x[0] = np.inf
    with test_util.use_gpu():
      with self.assertRaisesOpError(my_msg):
        t = constant_op.constant(x, shape=x_shape, dtype=dtypes.float32)
        t_verified = numerics.verify_tensor_all_finite(t, my_msg)
        self.evaluate(t_verified)


@test_util.run_v1_only("b/120545219")
class NumericsTest(test.TestCase):

  def testInf(self):
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant(1.0)
      t2 = constant_op.constant(0.0)
      a = math_ops.div(t1, t2)
      check = numerics.add_check_numerics_ops()
      a = control_flow_ops.with_dependencies([check], a)
      with self.assertRaisesOpError("Inf"):
        self.evaluate(a)

  def testNaN(self):
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant(0.0)
      t2 = constant_op.constant(0.0)
      a = math_ops.div(t1, t2)
      check = numerics.add_check_numerics_ops()
      a = control_flow_ops.with_dependencies([check], a)
      with self.assertRaisesOpError("NaN"):
        self.evaluate(a)

  def testBoth(self):
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant([1.0, 0.0])
      t2 = constant_op.constant([0.0, 0.0])
      a = math_ops.div(t1, t2)
      check = numerics.add_check_numerics_ops()
      a = control_flow_ops.with_dependencies([check], a)
      with self.assertRaisesOpError("Inf and NaN"):
        self.evaluate(a)

  def testPassThrough(self):
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
      checked = array_ops.check_numerics(t1, message="pass through test")
      value = self.evaluate(checked)
      self.assertAllEqual(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), value)
      self.assertEqual([2, 3], checked.get_shape())

  def testControlFlowCond(self):
    predicate = array_ops.placeholder(dtypes.bool, shape=[])
    _ = control_flow_ops.cond(predicate,
                              lambda: constant_op.constant([37.]),
                              lambda: constant_op.constant([42.]))
    with self.assertRaisesRegexp(
        ValueError,
        r"`tf\.add_check_numerics_ops\(\) is not compatible with "
        r"TensorFlow control flow operations such as `tf\.cond\(\)` "
        r"or `tf.while_loop\(\)`\."):
      numerics.add_check_numerics_ops()

  def testControlFlowWhile(self):
    predicate = array_ops.placeholder(dtypes.bool, shape=[])
    _ = control_flow_ops.while_loop(lambda _: predicate,
                                    lambda _: constant_op.constant([37.]),
                                    [constant_op.constant([42.])])
    with self.assertRaisesRegexp(
        ValueError,
        r"`tf\.add_check_numerics_ops\(\) is not compatible with "
        r"TensorFlow control flow operations such as `tf\.cond\(\)` "
        r"or `tf.while_loop\(\)`\."):
      numerics.add_check_numerics_ops()

  def testCheckNumericsV2OpNegativeAndPositveInf(self):
    """Test that CheckNumericsV2 op distinguishes negative and positive infs."""
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant([-1.0, 1.0])
      t2 = constant_op.constant([0.0, 0.0])
      checked = array_ops.check_numerics_v2(
          t1 / t2, message="pass through test")
      caught = None
      try:
        self.evaluate(checked)
      except errors.InvalidArgumentError as error:
        caught = error
      self.assertIn("had -Inf and +Inf values", caught.message)
      self.assertIn("pass through test", caught.message)

  def testCheckNumericsV2OpNegativeAndPositveInfAndNaN(self):
    """CheckNumericsV2 op distinguishes - & + infs when nan is present."""
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant([-1.0, 1.0, 0.0])
      t2 = constant_op.constant([0.0, 0.0, 0.0])
      checked = array_ops.check_numerics_v2(
          t1 / t2, message="pass through test")
      caught = None
      try:
        self.evaluate(checked)
      except errors.InvalidArgumentError as error:
        caught = error
      self.assertIn("had -Inf, +Inf, and NaN values", caught.message)
      self.assertIn("pass through test", caught.message)

  def testCheckNumericsV2PositveInfAndNaN(self):
    """Test that CheckNumericsV2 op shows sign of inf when nan is present."""
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant([0.0, 1.0])
      t2 = constant_op.constant([0.0, 0.0])
      checked = array_ops.check_numerics_v2(
          t1 / t2, message="pass through test")
      caught = None
      try:
        self.evaluate(checked)
      except errors.InvalidArgumentError as error:
        caught = error
      self.assertIn("had +Inf and NaN values", caught.message)
      self.assertIn("pass through test", caught.message)


if __name__ == "__main__":
  test.main()
=======
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        with self.assertRaisesOpError(my_msg):
          t = tf.constant(x, shape=x_shape, dtype=tf.float32)
          t_verified = tf.verify_tensor_all_finite(t, my_msg)
          t_verified.eval()

    # Test Inf.
    x[0] = np.inf
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        with self.assertRaisesOpError(my_msg):
          t = tf.constant(x, shape=x_shape, dtype=tf.float32)
          t_verified = tf.verify_tensor_all_finite(t, my_msg)
          t_verified.eval()


class NumericsTest(tf.test.TestCase):

  def testInf(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu, graph=tf.Graph()):
        t1 = tf.constant(1.0)
        t2 = tf.constant(0.0)
        a = tf.div(t1, t2)
        check = tf.add_check_numerics_ops()
        a = control_flow_ops.with_dependencies([check], a)
        with self.assertRaisesOpError("Inf"):
          a.eval()

  def testNaN(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu, graph=tf.Graph()):
        t1 = tf.constant(0.0)
        t2 = tf.constant(0.0)
        a = tf.div(t1, t2)
        check = tf.add_check_numerics_ops()
        a = control_flow_ops.with_dependencies([check], a)
        with self.assertRaisesOpError("NaN"):
          a.eval()

  def testBoth(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu, graph=tf.Graph()):
        t1 = tf.constant([1.0, 0.0])
        t2 = tf.constant([0.0, 0.0])
        a = tf.div(t1, t2)
        check = tf.add_check_numerics_ops()
        a = control_flow_ops.with_dependencies([check], a)
        with self.assertRaisesOpError("Inf and NaN"):
          a.eval()

  def testPassThrough(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu, graph=tf.Graph()):
        t1 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        checked = tf.check_numerics(t1, message="pass through test")
        value = checked.eval()
        self.assertAllEqual(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), value)
        self.assertEqual([2, 3], checked.get_shape())


if __name__ == "__main__":
  tf.test.main()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
