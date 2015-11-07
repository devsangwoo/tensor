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
"""Tests for tensorflow.ops.tf.BatchMatMul."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


def GetRandomNormalInput(shape, dtype):
  # float16 has limited range so we reduce the variance of the scalars.
  scale = 10.0 if dtype != np.float16 else 0.1
  loc = -10.0 if dtype != np.float16 else 0.1
  vals = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
  if dtype in (np.complex64, np.complex128):
    imag = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
    vals += 1j * imag
  return vals.reshape(shape)


class BatchMatmulOpTest(test.TestCase):

  # Uses numpy to compute batch_matmul(x, y, adjoint_a, adjoint_b).
  def _npBatchMatmul(self, x, y, adjoint_a, adjoint_b):
    # output's shape depends on adj[0] and adj[1]
    if adjoint_a:
      x = np.conjugate(np.swapaxes(x, -1, -2))
    if adjoint_b:
      y = np.conjugate(np.swapaxes(y, -1, -2))
    return np.matmul(x, y)

  # Compares TensorFlow BatchMatmul with NumPy's matmul.
  def _compare(self, x_in, y_in, adjoint_a, adjoint_b, static_shape):
    x_t_shape = x_in.shape[:-2] + (x_in.shape[-1], x_in.shape[-2])
    y_t_shape = y_in.shape[:-2] + (y_in.shape[-1], y_in.shape[-2])
    x = x_in if not adjoint_a else x_in.reshape(x_t_shape)
    y = y_in if not adjoint_b else y_in.reshape(y_t_shape)
    is_floating = x.dtype != np.int32
    tol = 100 * np.finfo(x.dtype).eps if is_floating else 0
    with self.cached_session(use_gpu=is_floating) as sess:
      if static_shape:
        z0 = math_ops.matmul(x, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
        z0_val = self.evaluate(z0)
      else:
        x_ph = array_ops.placeholder(x.dtype)
        y_ph = array_ops.placeholder(y.dtype)
        z0 = math_ops.matmul(
            x_ph, y_ph, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
        z0_val = sess.run(z0, feed_dict={x_ph: x, y_ph: y})
      z1 = self._npBatchMatmul(x, y, adjoint_a, adjoint_b)
      self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

  def _testNonEmpty(self, dtype, adjoint_a, adjoint_b, use_static_shape):

    def CompareNonEmpty(self, a_shape, b_shape):
      self._compare(
          GetRandomNormalInput(a_shape, dtype),
          GetRandomNormalInput(b_shape, dtype),
          adjoint_a,
          adjoint_b,
          static_shape=use_static_shape)

    CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [1, 3, 1])
    CompareNonEmpty(self, [1, 1, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
    CompareNonEmpty(self, [7, 1, 3], [7, 3, 5])
    CompareNonEmpty(self, [7, 2, 3], [7, 3, 1])
    CompareNonEmpty(self, [7, 2, 3], [7, 3, 5])
    CompareNonEmpty(self, [10, 64, 75], [10, 75, 30])
    CompareNonEmpty(self, [5, 7, 2, 3], [5, 7, 3, 5])

  def _testBroadcasting(self, dtype, adjoint_a, adjoint_b, use_static_shape):

    def CompareNonEmpty(self, a_shape, b_shape):
      self._compare(
          GetRandomNormalInput(a_shape, dtype),
          GetRandomNormalInput(b_shape, dtype),
          adjoint_a,
          adjoint_b,
          static_shape=use_static_shape)

    CompareNonEmpty(self, [2, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [3, 5])
    CompareNonEmpty(self, [5, 1, 2, 3], [1, 7, 3, 5])
    CompareNonEmpty(self, [5, 2, 2, 3], [3, 5])
    CompareNonEmpty(self, [2, 3], [5, 2, 3, 5])
    CompareNonEmpty(self, [4, 5, 1, 2, 3], [1, 1, 3, 5])
    CompareNonEmpty(self, [1, 2, 1, 4, 2, 1, 3, 4], [3, 2, 1, 1, 1, 2, 4, 2])

  def _testEmpty(self, dtype, adjoint_a, adjoint_b, use_static_shape):

    def CompareEmpty(self, a_shape, b_shape):
      self._compare(
          np.zeros(a_shape).astype(dtype),
          np.zeros(b_shape).astype(dtype),
          adjoint_a,
          adjoint_b,
          static_shape=use_static_shape)

    CompareEmpty(self, [0, 3, 2], [0, 2, 4])
    CompareEmpty(self, [3, 0, 2], [3, 2, 5])
    CompareEmpty(self, [3, 3, 2], [3, 2, 0])


def _GetBatchMatmulOpTest(dtype, adjoint_a, adjoint_b, use_static_shape):

  def Test(self):
    np.random.seed(42)
    self._testNonEmpty(dtype, adjoint_a, adjoint_b, use_static_shape)
    self._testEmpty(dtype, adjoint_a, adjoint_b, use_static_shape)

  return Test


def _GetBatchMatmulOpBroadcastingTest(dtype, adjoint_a, adjoint_b,
                                      use_static_shape):

  def Test(self):
    np.random.seed(42)
    self._testBroadcasting(dtype, adjoint_a, adjoint_b, use_static_shape)

  return Test


class BatchMatmulGradientTest(test.TestCase):

  # loss = sum(batch_matmul(x, y)). Verify dl/dx and dl/dy via the
  # gradient checker.
  def _checkGrad(self, x_in, y_in, adjoint_a, adjoint_b):
    x_t_shape = x_in.shape[:-2] + (x_in.shape[-1], x_in.shape[-2])
    y_t_shape = y_in.shape[:-2] + (y_in.shape[-1], y_in.shape[-2])
    x = x_in if not adjoint_a else x_in.reshape(x_t_shape)
    y = y_in if not adjoint_b else y_in.reshape(y_t_shape)
    epsilon = np.finfo(x.dtype).eps
    # Since our gradient is linear, a larger delta decreases the error.
    delta = 10 * epsilon**(1.0 / 3.0)

    def Loss(x, y):
      return math_ops.reduce_sum(math_ops.matmul(x, y, adjoint_a, adjoint_b))

    with self.cached_session(use_gpu=True):
      ((x_jacob_t, y_jacob_t),
       (x_jacob_n, y_jacob_n)) = gradient_checker_v2.compute_gradient(
           Loss, [x, y], delta=delta)
      tol = 10 * delta
      self.assertAllClose(x_jacob_t, x_jacob_n, rtol=tol, atol=tol)
      self.assertAllClose(y_jacob_t, y_jacob_n, rtol=tol, atol=tol)

  # Tests gradients of a batched matmul of x, and y
  def _compare(self, a_shape, b_shape, dtype, adjoint_a, adjoint_b):
    np.random.seed(42)
    x = GetRandomNormalInput(a_shape, dtype)
    y = GetRandomNormalInput(b_shape, dtype)
    self._checkGrad(x, y, adjoint_a, adjoint_b)


def _GetBatchMatmulGradientTest(dtype, adjoint_a, adjoint_b):

  def Test(self):
    def CheckGradients(self, a_shape, b_shape):
      self._compare(a_shape, b_shape, dtype, adjoint_a, adjoint_b)

    CheckGradients(self, [1, 2, 3], [1, 3, 5])
    CheckGradients(self, [3, 4, 7], [3, 7, 10])

  return Test


def _GetBatchMatmulGradientWithBroadcastingTest(dtype, adjoint_a, adjoint_b):

  def Test(self):
    def CheckGradients(self, a_shape, b_shape):
      self._compare(a_shape, b_shape, dtype, adjoint_a, adjoint_b)

    CheckGradients(self, [1, 5, 2, 3], [7, 1, 3, 2])
    CheckGradients(self, [2, 3], [1, 3, 5])
    CheckGradients(self, [2, 3], [5, 3, 5])
    CheckGradients(self, [5, 2, 5], [5, 3])
    CheckGradients(self, [5, 2, 2, 3], [3, 5])
    CheckGradients(self, [4, 5, 1, 2, 3], [1, 1, 3, 5])
    CheckGradients(self, [1, 2, 1, 4, 2, 1, 3, 4], [3, 2, 1, 1, 1, 2, 4, 2])

  return Test


class BatchMatMulBenchmark(test.Benchmark):
  # Batch sizes are 512.
  shape_pairs = [
      # Typical fully connected layer.
      ((4, 8, 4, 2, 1, 1024), (1024, 1024)),
      ((4, 1, 4, 1, 1, 1024), (1, 8, 1, 2, 1024, 1024)),
      # Square matmul.
      ((4, 8, 4, 2, 512, 512), (512, 512)),
      ((4, 1, 4, 1, 512, 512), (1, 8, 1, 2, 512, 512)),
      # Matrix-vector multiplies.
      ((4, 8, 4, 2, 10000, 200), (200, 1)),
      ((4, 1, 4, 1, 10000, 200), (1, 8, 1, 2, 200, 1)),
      # Vector-matrix multiplies.
      ((4, 8, 4, 2, 1, 200), (200, 10000)),
      ((4, 1, 4, 1, 1, 200), (1, 8, 1, 2, 200, 10000)),
  ]

  def benchmarkBatchMatMulBroadcast(self):
    for (a_shape, b_shape) in self.shape_pairs:
      with ops.Graph().as_default(), \
          session.Session(config=benchmark.benchmark_config()) as sess, \
          ops.device("/cpu:0"):
        matrix_a = variables.Variable(
            GetRandomNormalInput(a_shape, np.float32))
        matrix_b = variables.Variable(
            GetRandomNormalInput(b_shape, np.float32))
        variables.global_variables_initializer().run()

        # Use batch matmul op's internal broadcasting.
        self.run_op_benchmark(
            sess,
            math_ops.matmul(matrix_a, matrix_b),
            min_iters=50,
            name="batch_matmul_cpu_{}_{}".format(a_shape, b_shape))

        # Manually broadcast the input matrices using the broadcast_to op.
        broadcasted_batch_shape = array_ops.broadcast_static_shape(
            matrix_a.shape[:-2], matrix_b.shape[:-2])
        broadcasted_a_shape = broadcasted_batch_shape.concatenate(
            matrix_a.shape[-2:])
        broadcasted_b_shape = broadcasted_batch_shape.concatenate(
            matrix_b.shape[-2:])
        self.run_op_benchmark(
            sess,
            math_ops.matmul(
                array_ops.broadcast_to(matrix_a, broadcasted_a_shape),
                array_ops.broadcast_to(matrix_b, broadcasted_b_shape)),
            min_iters=50,
            name="batch_matmul_manual_broadcast_cpu_{}_{}".format(
                a_shape, b_shape))


if __name__ == "__main__":
  dtypes_to_test = [np.float16, np.float32, np.float64, np.int32]
  if not test.is_built_with_rocm():
    # ROCm does not support BLAS operations for complex types
    dtypes_to_test += [np.complex64, np.complex128]
  for dtype_ in dtypes_to_test:
    for adjoint_a_ in False, True:
      for adjoint_b_ in False, True:
        name = "%s_%s_%s" % (dtype_.__name__, adjoint_a_, adjoint_b_)
        # TF2 does not support placeholders under eager so we skip it.
        for use_static_shape_ in set([True, tf2.enabled()]):
          setattr(
              BatchMatmulOpTest,
              "testBatchMatmulOp_" + name + "_{}".format(use_static_shape_),
              test_util.xla_allow_fallback(
                  "TODO(b/134526360): XLA:CPU hasn't implemented int32 dot.")(
                      _GetBatchMatmulOpTest(dtype_, adjoint_a_, adjoint_b_,
                                            use_static_shape_)))
          # Broadcasting is supported only in v2.
          setattr(
              BatchMatmulOpTest, "testBatchMatmulBroadcasting_" + name +
              ("_%s" % use_static_shape_),
              test_util.xla_allow_fallback(
                  "TODO(b/134526360): XLA:CPU hasn't implemented int32 dot.")(
                      _GetBatchMatmulOpBroadcastingTest(dtype_, adjoint_a_,
                                                        adjoint_b_,
                                                        use_static_shape_)))
        if dtype_ == np.int32:
          continue
        setattr(BatchMatmulGradientTest, "testBatchMatmulGradient_" + name,
                _GetBatchMatmulGradientTest(dtype_, adjoint_a_, adjoint_b_))
        # Broadcasting is supported only in v2.
        setattr(
            BatchMatmulGradientTest,
            "testBatchMatmulGradientWithBroadcasting_" + name,
            _GetBatchMatmulGradientWithBroadcastingTest(dtype_, adjoint_a_,
                                                        adjoint_b_))
  test.main()
=======
"""Tests for tensorflow.ops.tf.BatchMatMul."""

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker as gc


class BatchMatmulOpTest(tf.test.TestCase):

  # Uses numpy to compute batch_matmul(x, y, adj_x, adj_y).
  def _npBatchMatmul(self, x, y, adj_x, adj_y):
    assert x.ndim >= 3
    assert y.ndim >= 3
    # output's shape depends on adj[0] and adj[1]
    d0 = x.shape[-2] if not adj_x else x.shape[-1]
    d2 = y.shape[-1] if not adj_y else y.shape[-2]
    batch_dims = x.shape[:-2]
    num = np.prod(batch_dims)
    z = np.empty(list(batch_dims) + [d0, d2], dtype=x.dtype)
    xr = x.reshape([num, x.shape[-2], x.shape[-1]])
    yr = y.reshape([num, y.shape[-2], y.shape[-1]])
    zr = z.reshape([num, z.shape[-2], z.shape[-1]])
    for i in range(num):
      a = np.matrix(xr[i, :, :])
      if adj_x:
        a = a.transpose().conj()
      b = np.matrix(yr[i, :, :])
      if adj_y:
        b = b.transpose().conj()
      zr[i, :, :] = a * b
    return z

  # Test _npBatchMatMul works.
  def testSimpleNpVersion(self):
    x = np.array([0., 1., 2., 3.]).reshape([1, 2, 2])
    y = np.array([1., 2., 3., 4.]).reshape([1, 2, 2])
    z0 = self._npBatchMatmul(x, y, False, False)
    z1 = np.array([3., 4., 11., 16.]).reshape([1, 2, 2])
    self.assertTrue(np.array_equal(z0, z1))

    x = np.array([1., (1j), (-1.), (-1j)]).reshape([1, 2, 2])
    y = x * np.complex(1, 1)  # rotate x 90 degree
    z0 = self._npBatchMatmul(x, y, False, False)
    z1 = np.array([2., (2.j), -2., (-2.j)]).reshape([1, 2, 2])
    self.assertTrue(np.array_equal(z0, z1))

    z0 = self._npBatchMatmul(x, y, False, True)
    z1 = np.array([(2.-2.j), (-2.+2.j), (-2.+2.j), (2.-2.j)]).reshape([1, 2, 2])
    self.assertTrue(np.array_equal(z0, z1))

    z0 = self._npBatchMatmul(x, y, True, False)
    z1 = np.array([(2.+2.j), (-2.+2.j), (2.-2.j), (2.+2.j)]).reshape([1, 2, 2])
    self.assertTrue(np.array_equal(z0, z1))

  # Compares _tfpBatchMatmul(x, y, alpha, adj) and _npBatchMatMul(x, y, alpha,
  # adj)
  def _compare(self, x, y, adj_x, adj_y, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      z0 = tf.batch_matmul(x, y, adj_x=adj_x, adj_y=adj_y)
      z0_val = z0.eval()
    z1 = self._npBatchMatmul(x, y, adj_x, adj_y)
    self.assertShapeEqual(z1, z0)
    if z0_val.size != 0:
      err = (np.abs(z0_val - z1) / np.maximum(1, np.abs(z0_val))).max()
      tf.logging.info("error = %f", err)
      self.assertTrue(err < 1e-4)

  # Returns a random float np of "shape".
  def _randFloat(self, shape):
    vals = np.random.normal(0, 1, np.prod(shape)).reshape(shape)
    return np.array(vals, dtype=np.float32)

  def testSimpleFloat(self):
    for use_gpu in [False, True]:
      self._compare(self._randFloat([7, 2, 3]), self._randFloat([7, 3, 5]),
                    False, False, use_gpu)
      self._compare(self._randFloat([7, 2, 3]), self._randFloat([7, 5, 3]),
                    False, True, use_gpu)
      self._compare(self._randFloat([7, 3, 2]), self._randFloat([7, 3, 5]),
                    True, False, use_gpu)
      self._compare(self._randFloat([7, 3, 2]), self._randFloat([7, 5, 3]),
                    True, True, use_gpu)

  def testLargeFloat(self):
    for use_gpu in [False, True]:
      self._compare(self._randFloat([10, 64, 75]),
                    self._randFloat([10, 75, 30]), False, False, use_gpu)
      self._compare(self._randFloat([10, 75, 64]),
                    self._randFloat([10, 75, 30]), True, False, use_gpu)
      self._compare(self._randFloat([10, 64, 75]),
                    self._randFloat([10, 30, 75]), False, True, use_gpu)
      self._compare(self._randFloat([10, 75, 64]),
                    self._randFloat([10, 30, 75]), True, True, use_gpu)

  def testHighNDims(self):
    for use_gpu in [False, True]:
      self._compare(self._randFloat([5, 7, 2, 3]),
                    self._randFloat([5, 7, 3, 5]), False, False, use_gpu)
      self._compare(self._randFloat([5, 7, 3, 2]),
                    self._randFloat([5, 7, 3, 5]), True, False, use_gpu)
      self._compare(self._randFloat([5, 7, 2, 3]),
                    self._randFloat([5, 7, 5, 3]), False, True, use_gpu)
      self._compare(self._randFloat([5, 7, 3, 2]),
                    self._randFloat([5, 7, 5, 3]), True, True, use_gpu)

  # Returns a random complex numpy array of "shape".
  def _randComplex(self, shape):
    real = np.random.normal(0, 1, np.prod(shape))
    imag = np.random.normal(0, 1, np.prod(shape))
    vals = [np.complex(v[0], v[1]) for v in zip(real, imag)]
    return np.array(vals, dtype=np.complex64).reshape(shape)

  def testSimpleComplex(self):
    self._compare(self._randComplex([7, 2, 3]),
                  self._randComplex([7, 3, 5]), False, False)
    self._compare(self._randComplex([7, 2, 3]),
                  self._randComplex([7, 5, 3]), False, True)
    self._compare(self._randComplex([7, 3, 2]),
                  self._randComplex([7, 3, 5]), True, False)
    self._compare(self._randComplex([7, 3, 2]),
                  self._randComplex([7, 5, 3]), True, True)

  def testLargeComplex(self):
    self._compare(self._randComplex([10, 64, 75]),
                  self._randComplex([10, 75, 30]), False,
                  False)
    self._compare(self._randComplex([10, 64, 75]),
                  self._randComplex([10, 30, 75]), False, True)
    self._compare(self._randComplex([10, 75, 64]),
                  self._randComplex([10, 75, 30]), True, False)
    self._compare(self._randComplex([10, 75, 64]),
                  self._randComplex([10, 30, 75]), True, True)

  def testEmpty(self):
    self._compare(np.empty([0, 3, 2]).astype(np.float32),
                  np.empty([0, 2, 4]).astype(np.float32), False, False)
    self._compare(np.empty([3, 2, 0]).astype(np.float32),
                  np.empty([3, 0, 5]).astype(np.float32), False, False)
    self._compare(np.empty([3, 0, 2]).astype(np.float32),
                  np.empty([3, 2, 5]).astype(np.float32), False, False)
    self._compare(np.empty([3, 3, 2]).astype(np.float32),
                  np.empty([3, 2, 0]).astype(np.float32), False, False)


class BatchMatmulGradientTest(tf.test.TestCase):

  # loss = sum(batch_matmul(x, y)). Verify dl/dx and dl/dy via the
  # gradient checker.
  def _checkGrad(self, x, y, adj_x, adj_y):
    assert 3 == x.ndim
    assert 3 == y.ndim
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      z = tf.batch_matmul(inx, iny, adj_x, adj_y)
      loss = tf.reduce_sum(z)
      epsilon = 1e-2
      ((x_jacob_t, x_jacob_n), (y_jacob_t, y_jacob_n)) = gc.ComputeGradient(
          [inx, iny], [x.shape, y.shape], loss, [1],
          x_init_value=[x, y], delta=epsilon)

    tf.logging.info("x_jacob_t = %s", x_jacob_t.reshape(x.shape))
    tf.logging.info("x_jacob_n = %s", x_jacob_n.reshape(x.shape))
    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)
    tf.logging.info("y_jacob_t = %s", y_jacob_t.reshape(y.shape))
    tf.logging.info("y_jacob_n = %s", y_jacob_n.reshape(y.shape))
    self.assertAllClose(y_jacob_t, y_jacob_n, rtol=1e-2, atol=epsilon)

  # Tests a batched matmul of x, and y: x is a 3D tensor of shape [b,
  # n, k] y is a 3D tensor of shape [b, k, m] the batched matmul
  # computes z of shape [b, n, m], where z[i, :, :] = x[i, :, :]
  # matmul y[i, :, :]
  def _compare(self, b, n, k, m):
    x = np.random.normal(0, 1, b * n * k).astype(np.float32).reshape([b, n, k])
    y = np.random.normal(0, 1, b * k * m).astype(np.float32).reshape([b, k, m])
    self._checkGrad(x, y, False, False)
    self._checkGrad(x.reshape([b, k, n]), y, True, False)
    self._checkGrad(x, y.reshape([b, m, k]), False, True)
    self._checkGrad(x.reshape([b, k, n]), y.reshape([b, m, k]), True, True)

  def testSmall(self):
    self._compare(1, 2, 3, 5)

  def testMedium(self):
    self._compare(3, 4, 7, 10)

  # Can't do testLarge using very large inputs because gradient
  # checker will take way too long time.


if __name__ == "__main__":
  tf.test.main()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
