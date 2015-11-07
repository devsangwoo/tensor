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
"""Tests for Adam."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam


def adam_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      alpha=0.001,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-8):
  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)
=======
"""Tests for Adam."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


def adam_update_numpy(param, g_t, t, m, v, alpha=0.001, beta1=0.9, beta2=0.999,
                      epsilon=1e-8):
  alpha_t = alpha * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


<<<<<<< HEAD
class AdamOptimizerTest(test.TestCase):

  def doTestSparse(self, use_resource=False):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(var0_np)
          var1 = resource_variable_ops.ResourceVariable(var1_np)
        else:
          var0 = variables.RefVariable(var0_np)
          var1 = variables.RefVariable(var1_np)
        grads0_np_indices = np.array([0, 1], dtype=np.int32)
        grads0 = ops.IndexedSlices(
            constant_op.constant(grads0_np),
            constant_op.constant(grads0_np_indices), constant_op.constant([2]))
        grads1_np_indices = np.array([0, 1], dtype=np.int32)
        grads1 = ops.IndexedSlices(
            constant_op.constant(grads1_np),
            constant_op.constant(grads1_np_indices), constant_op.constant([2]))
        opt = adam.AdamOptimizer()
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**t,
                                             self.evaluate(beta2_power))
          update.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testSparse(self):
    self.doTestSparse(use_resource=False)

  @test_util.run_deprecated_v1
  def testResourceSparse(self):
    self.doTestSparse(use_resource=True)

  @test_util.run_deprecated_v1
  def testSparseDevicePlacement(self):
    for index_dtype in [dtypes.int32, dtypes.int64]:
      with self.cached_session(force_gpu=test.is_gpu_available()):
        # If a GPU is available, tests that all optimizer ops can be placed on
        # it (i.e. they have GPU kernels).
        var = variables.Variable([[1.0], [2.0]])
        indices = constant_op.constant([0, 1], dtype=index_dtype)
        gathered_sum = math_ops.reduce_sum(array_ops.gather(var, indices))
        optimizer = adam.AdamOptimizer(3.0)
        minimize_op = optimizer.minimize(gathered_sum)
        variables.global_variables_initializer().run()
        minimize_op.run()

  @test_util.run_deprecated_v1
  def testSparseRepeatedIndices(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        repeated_index_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        aggregated_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        grad_repeated_index = ops.IndexedSlices(
            constant_op.constant(
                [0.1, 0.1], shape=[2, 1], dtype=dtype),
            constant_op.constant([1, 1]),
            constant_op.constant([2, 1]))
        grad_aggregated = ops.IndexedSlices(
            constant_op.constant(
                [0.2], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]),
            constant_op.constant([2, 1]))
        repeated_update = adam.AdamOptimizer().apply_gradients(
            [(grad_repeated_index, repeated_index_update_var)])
        aggregated_update = adam.AdamOptimizer().apply_gradients(
            [(grad_aggregated, aggregated_update_var)])
        variables.global_variables_initializer().run()
        self.assertAllClose(aggregated_update_var.eval(),
                            self.evaluate(repeated_index_update_var))
        for _ in range(3):
          repeated_update.run()
          aggregated_update.run()
          self.assertAllClose(aggregated_update_var.eval(),
                              self.evaluate(repeated_index_update_var))

  def doTestBasic(self, use_resource=False, use_callable_params=False):
    if context.executing_eagerly() and not use_resource:
      self.skipTest(
          "Skipping test with use_resource=False and executing eagerly.")
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.session(graph=ops.Graph()):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(
              var0_np, name="var0_%d" % i)
          var1 = resource_variable_ops.ResourceVariable(
              var1_np, name="var1_%d" % i)
        else:
          var0 = variables.RefVariable(var0_np)
          var1 = variables.RefVariable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        learning_rate = lambda: 0.001
        beta1 = lambda: 0.9
        beta2 = lambda: 0.999
        epsilon = lambda: 1e-8
        if not use_callable_params:
          learning_rate = learning_rate()
          beta1 = beta1()
          beta2 = beta2()
          epsilon = epsilon()

        opt = adam.AdamOptimizer(learning_rate=learning_rate)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        opt_variables = opt.variables()
        beta1_power, beta2_power = opt._get_beta_accumulators()
        self.assertTrue(beta1_power is not None)
        self.assertTrue(beta2_power is not None)
        self.assertIn(beta1_power, opt_variables)
        self.assertIn(beta2_power, opt_variables)
        # Ensure that non-slot variables are the same type as the requested
        # variables.
        self.assertEqual(
            use_resource,
            resource_variable_ops.is_resource_variable(beta1_power))
        self.assertEqual(
            use_resource,
            resource_variable_ops.is_resource_variable(beta2_power))

        if not context.executing_eagerly():
          with ops.Graph().as_default():
            # Shouldn't return non-slot variables from other graphs.
            self.assertEqual(0, len(opt.variables()))
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          if not context.executing_eagerly():
            self.evaluate(update)
          elif t > 1:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          self.assertAllCloseAccordingToType(0.9**(t + 1),
                                             self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**(t + 1),
                                             self.evaluate(beta2_power))

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))
          if use_resource:
            self.assertEqual("var0_%d/Adam:0" % (i,),
                             opt.get_slot(var=var0, name="m").name)

  def testBasic(self):
    with self.cached_session():
      self.doTestBasic(use_resource=False)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)

  def testBasicCallableParams(self):
    with context.eager_mode():
      self.doTestBasic(use_resource=True, use_callable_params=True)

  @test_util.run_deprecated_v1
  def testTensorLearningRate(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = adam.AdamOptimizer(constant_op.constant(0.001))
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**t,
                                             self.evaluate(beta2_power))
          update.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testSharing(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = adam.AdamOptimizer()
        update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 3 steps of intertwined Adam1 and Adam2.
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**t,
                                             self.evaluate(beta2_power))
          if t % 2 == 0:
            update1.run()
          else:
            update2.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testTwoSessions(self):
    optimizer = adam.AdamOptimizer()

    with context.eager_mode():
      var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
      grads0 = constant_op.constant(np.array([0.1, 0.1]))
      optimizer.apply_gradients([(grads0, var0)])

    g = ops.Graph()
    with g.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        grads0 = constant_op.constant(np.array([0.1, 0.1]))
        optimizer.apply_gradients([(grads0, var0)])

    gg = ops.Graph()
    with gg.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        grads0 = constant_op.constant(np.array([0.1, 0.1]))

        # If the optimizer saves any state not keyed by graph the following line
        # fails.
        optimizer.apply_gradients([(grads0, var0)])

  def testSlotsUniqueEager(self):
    with context.eager_mode():
      v1 = resource_variable_ops.ResourceVariable(1.)
      v2 = resource_variable_ops.ResourceVariable(1.)
      opt = adam.AdamOptimizer(1.)
      opt.minimize(lambda: v1 + v2)
      # There should be two non-slot variables, and two unique slot variables
      # for v1 and v2 respectively.
      self.assertEqual(6, len({id(v) for v in opt.variables()}))


if __name__ == "__main__":
  test.main()
=======
class AdamOptimizerTest(tf.test.TestCase):

  def testSparse(self):
    with self.test_session():
      # Initialize variables for numpy implementation.
      m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
      var0_np = np.array([1.0, 2.0], dtype=np.float32)
      grads0_np = np.array([0.1, 0.1], dtype=np.float32)
      var1_np = np.array([3.0, 4.0], dtype=np.float32)
      grads1_np = np.array([0.01, 0.01], dtype=np.float32)

      var0 = tf.Variable(var0_np)
      var1 = tf.Variable(var1_np)
      grads0_np_indices = np.array([0, 1], dtype=np.int32)
      grads0 = tf.IndexedSlices(tf.constant(grads0_np),
                                tf.constant(grads0_np_indices),
                                tf.constant([2]))
      grads1_np_indices = np.array([0, 1], dtype=np.int32)
      grads1 = tf.IndexedSlices(tf.constant(grads1_np),
                                tf.constant(grads1_np_indices),
                                tf.constant([2]))
      opt = tf.train.AdamOptimizer()
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      tf.initialize_all_variables().run()

      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())

      beta1_power, beta2_power = opt._get_beta_accumulators()

      # Run 3 steps of Adam
      for t in range(1, 4):
        self.assertAllClose(0.9 ** t, beta1_power.eval())
        self.assertAllClose(0.999 ** t, beta2_power.eval())
        update.run()

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        self.assertAllClose(var0_np, var0.eval())
        self.assertAllClose(var1_np, var1.eval())

  def testBasic(self):
    with self.test_session():
      # Initialize variables for numpy implementation.
      m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
      var0_np = np.array([1.0, 2.0], dtype=np.float32)
      grads0_np = np.array([0.1, 0.1], dtype=np.float32)
      var1_np = np.array([3.0, 4.0], dtype=np.float32)
      grads1_np = np.array([0.01, 0.01], dtype=np.float32)

      var0 = tf.Variable(var0_np)
      var1 = tf.Variable(var1_np)
      grads0 = tf.constant(grads0_np)
      grads1 = tf.constant(grads1_np)
      opt = tf.train.AdamOptimizer()
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      tf.initialize_all_variables().run()

      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())

      beta1_power, beta2_power = opt._get_beta_accumulators()

      # Run 3 steps of Adam
      for t in range(1, 4):
        self.assertAllClose(0.9 ** t, beta1_power.eval())
        self.assertAllClose(0.999 ** t, beta2_power.eval())
        update.run()

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        self.assertAllClose(var0_np, var0.eval())
        self.assertAllClose(var1_np, var1.eval())

  def testFloat64(self):
    with self.test_session():
      opt = tf.train.AdamOptimizer()

      # compute_gradients.
      values = [1.0, 3.0]
      good_vars = [tf.Variable([v]) for v in values]
      bad_loss = tf.constant(2.0, tf.float64, name="bad_loss")
      self.assertRaisesRegexp(
          ValueError, r"Invalid type.*float64.*bad_loss.*expected.*float32",
          opt.compute_gradients, bad_loss, good_vars)
      bad_vars = [
          tf.Variable(np.array([v], np.float64), name="bad_var")
          for v in values]
      self.assertRaisesRegexp(
          ValueError, r"Invalid type.*float64.*bad_var.*expected.*float32",
          opt.compute_gradients, tf.cast(bad_vars[0] + bad_vars[1], tf.float32),
          bad_vars)
      opt.compute_gradients(good_vars[0] + good_vars[1], good_vars)

      # apply_gradients.
      bad_grads = [
          tf.constant([0.1], dtype=np.float64, name="bad_grad"),
          tf.constant([0.01])]
      self.assertRaisesRegexp(
          ValueError, r"Invalid type.*float64.*bad_grad.*expected.*float32",
          opt.apply_gradients, zip(bad_grads, good_vars))
      good_grads = [tf.constant([0.01]), tf.constant([0.02])]
      self.assertRaisesRegexp(
          ValueError, r"Invalid type.*float64.*bad_var.*expected.*float32",
          opt.apply_gradients, zip(good_grads, bad_vars))
      opt.apply_gradients(zip(good_grads, good_vars))

  def testSharing(self):
    with self.test_session():
      # Initialize variables for numpy implementation.
      m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
      var0_np = np.array([1.0, 2.0], dtype=np.float32)
      grads0_np = np.array([0.1, 0.1], dtype=np.float32)
      var1_np = np.array([3.0, 4.0], dtype=np.float32)
      grads1_np = np.array([0.01, 0.01], dtype=np.float32)

      var0 = tf.Variable(var0_np)
      var1 = tf.Variable(var1_np)
      grads0 = tf.constant(grads0_np)
      grads1 = tf.constant(grads1_np)
      opt = tf.train.AdamOptimizer()
      update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      tf.initialize_all_variables().run()

      beta1_power, beta2_power = opt._get_beta_accumulators()

      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())

      # Run 3 steps of intertwined Adam1 and Adam2.
      for t in range(1, 4):
        self.assertAllClose(0.9 ** t, beta1_power.eval())
        self.assertAllClose(0.999 ** t, beta2_power.eval())
        if t % 2 == 0:
          update1.run()
        else:
          update2.run()

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        self.assertAllClose(var0_np, var0.eval())
        self.assertAllClose(var1_np, var1.eval())


if __name__ == "__main__":
  tf.test.main()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
