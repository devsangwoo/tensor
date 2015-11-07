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
"""Tests for tensorflow.ops.math_ops.matmul."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as test_lib

# TODO(yangzihao): Currently matmul autotuning is disabled by default. Use
# os.environ["TF_MATMUL_AUTOTUNE_ENABLE"] = "1" to enable it.


class MatVecTest(test_lib.TestCase):
  """Simple test for matvec, which is sugar on top of matmul."""

  def testTwoByTwoCase(self):
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    c = math_ops.matvec(a, b)
    self.assertAllEqual((2,), c.shape)
    self.assertAllEqual([5 + 2 * 6, 3 * 5 + 4 * 6], c)


def _AddTest(test, op_name, testcase_name, fn):
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, test_util.deprecated_graph_mode_only(fn))


def _GetTransposedMatrices(x, x_name, kwargs):
  if kwargs["transpose_" + x_name] is True:
    return x.T
  elif kwargs["adjoint_" + x_name] is True:
    return np.conj(x.T)
  else:
    return x


class MatMulTest(test_lib.TestCase):
  pass  # Filled in below


def _GetMatMulTest(a_np_, b_np_, use_static_shape_, **kwargs_):

  def Test(self):
    np_val = np.matrix(a_np_) * np.matrix(b_np_)

    use_gpu = True
    if a_np_.dtype is np.float16 and (
        not test_util.GpuSupportsHalfMatMulAndConv()):
      use_gpu = False
      print("Built without fp16 matmul support for Cuda, running test on CPU.")

    # Transpose and possibly conjugate a_np_ and b_np_ according to the
    # attributes such that tf.matmul(effective_a_np, effective_b_np, **kwargs)
    # results in a valid matrix multiplication and produces the same result as
    # np.matrix(a_np_) * np.matrix(b_np_)
    effective_a_np = _GetTransposedMatrices(a_np_, "a", kwargs_)
    effective_b_np = _GetTransposedMatrices(b_np_, "b", kwargs_)
    with self.cached_session() as sess, test_util.device(use_gpu):
      if use_static_shape_:
        a = constant_op.constant(effective_a_np)
        b = constant_op.constant(effective_b_np)
        res = math_ops.matmul(a, b, **kwargs_)
        tf_val = self.evaluate(res)
      else:
        a = array_ops.placeholder(a_np_.dtype)
        b = array_ops.placeholder(b_np_.dtype)
        res = math_ops.matmul(a, b, **kwargs_)
        tf_val = sess.run(res, feed_dict={a: effective_a_np, b: effective_b_np})

    self.assertAllCloseAccordingToType(
        tf_val,
        np_val,
        float_rtol=3e-5,
        float_atol=3e-5,
        half_rtol=0.2,
        half_atol=0.2)

  return Test


class MatMulGradientTest(test_lib.TestCase):
  pass  # Will be filled in below.


def _GetMatMulGradientTest(a_np_, b_np_, use_static_shape_, **kwargs_):

  def Test(self):
    if not use_static_shape_ or a_np_.dtype in (np.int32, np.int64, np.float16):
      self.skipTest("Skipping infeasible gradient test.")

    # Transpose and possibly conjugate a_np_ and b_np_ according to the
    # attributes such that tf.matmul(effective_a_np, effective_b_np, **kwargs)
    # results in a valid matrix multiplication and produces the same result as
    # np.matrix(a_np_) * np.matrix(b_np_)
    effective_a_np = _GetTransposedMatrices(a_np_, "a", kwargs_)
    effective_b_np = _GetTransposedMatrices(b_np_, "b", kwargs_)

    epsilon = np.finfo(a_np_.dtype).eps
    delta = epsilon**(1.0 / 3.0)
    tol = 20 * delta
    with self.session():
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.matmul(x, effective_b_np, **kwargs_),
          [effective_a_np],
          delta=delta)
      self.assertAllClose(theoretical, numerical, rtol=tol, atol=tol)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.matmul(effective_a_np, x, **kwargs_),
          [effective_b_np],
          delta=delta)
      self.assertAllClose(theoretical, numerical, rtol=tol, atol=tol)

  return Test


class MatMulStatsTest(test_lib.TestCase):

  @test_util.run_v1_only("Test requires a Graph and NodeDef inspection")
  def testSimpleStatistics(self):
    a = variables.Variable(random_ops.random_normal([25, 16]))
    b = variables.Variable(random_ops.random_normal([16, 9]))
    math_ops.matmul(a, b)
    g = ops.get_default_graph()
    for op in g.get_operations():
      flops = ops.get_stats_for_node_def(g, op.node_def, "flops").value
      if op.name == "MatMul":
        self.assertEqual(7200, flops)

  @test_util.run_v1_only("Test requires a Graph and NodeDef inspection")
  def testTransposedStatistics(self):
    a = variables.Variable(random_ops.random_normal([16, 25]))
    b = variables.Variable(random_ops.random_normal([16, 9]))
    math_ops.matmul(a, b, transpose_a=True)
    g = ops.get_default_graph()
    for op in g.get_operations():
      flops = ops.get_stats_for_node_def(g, op.node_def, "flops").value
      if op.name == "MatMul":
        self.assertEqual(7200, flops)


try:
  # @ operator supported since python 3.5.
  infix_matmul = operator.matmul
except AttributeError:

  # For earlier versions of python, emulate regular behavior.
  # Useful to build and test for 3.5+ on earlier versions.
  def infix_matmul(x, y):  # pylint: disable=invalid-name
    try:
      r = type(x).__matmul__(x, y)
    except AttributeError:
      r = NotImplemented
    if r is NotImplemented and type(x) is not type(y):
      try:
        r = type(y).__rmatmul__(y, x)
      except AttributeError:
        r = NotImplemented
    if r is NotImplemented:
      raise TypeError("unsupported operand type(s) for @: '{}' and '{}'"
                      .format(type(x).__name__, type(y).__name__))
    return r


class MatMulInfixOperatorTest(test_lib.TestCase):

  def testMismatchedShape(self):
    with self.assertRaisesRegexp(
        Exception, "(Shape must be rank 2 but is rank 1|is not a matrix)"):
      infix_matmul(
          ops.convert_to_tensor([10.0, 20.0, 30.0]),
          ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0]]))

  def testMismatchedDimensions(self):
    with self.assertRaisesRegexp(
        Exception, "(Dimensions must be equal|Matrix size-incompatible)"):
      infix_matmul(
          ops.convert_to_tensor([[10.0, 20.0, 30.0]]),
          ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0]]))

  @test_util.run_v1_only("Tensor.op is generally not applicable in TF 2")
  def testInfixMatmulIsTfMatmul(self):
    a = ops.convert_to_tensor([[10.0, 20.0, 30.0]])
    b = ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0], [80.0, 90.0]])
    c = infix_matmul(a, b)
    self.assertEqual(c.op.type, "MatMul")

  def testInfixMatmulDoesDotProduct(self):
    a = ops.convert_to_tensor([[10.0, 20.0, 30.0]])
    b = ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0], [80.0, 90.0]])
    c = infix_matmul(a, b)
    d = math_ops.matmul(a, b)
    self.assertAllEqual(c, d)


if __name__ == "__main__":
  sizes = [1, 3, 5]
  trans_options = [[False, False], [True, False], [False, True]]
  dtypes_to_test = [np.int32, np.int64, np.float16, np.float32, np.float64]
  if not test_lib.is_built_with_rocm():
    # ROCm does not support BLAS operations for complex types
    dtypes_to_test += [np.complex64, np.complex128]
  # TF2 does not support placeholders under eager so we skip it
  for use_static_shape in set([True, tf2.enabled()]):
    for dtype in dtypes_to_test:
      if not use_static_shape and (dtype == np.int32 or dtype == np.int64):
        # TODO(rmlarsen): Re-enable this test when we have fixed the underlying
        # bug in Windows (b/35935459).
        continue
      for m in sizes:
        for n in sizes:
          for k in sizes:
            # Construct compatible random matrices a_np of size [m, k] and b_np
            # of size [k, n].
            a_np = np.random.normal(-5, 5, m * k).astype(dtype).reshape([m, k])
            if dtype in (np.complex64, np.complex128):
              a_np.imag = np.random.normal(-5, 5,
                                           m * k).astype(dtype).reshape([m, k])
            b_np = np.random.normal(-5, 5, k * n).astype(dtype).reshape([k, n])
            if dtype in (np.complex64, np.complex128):
              b_np.imag = np.random.normal(-5, 5,
                                           k * n).astype(dtype).reshape([k, n])
            for adjoint_a, transpose_a in trans_options:
              for adjoint_b, transpose_b in trans_options:
                name = "%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                    use_static_shape, dtype.__name__, m, n, k, adjoint_a,
                    transpose_a, adjoint_b, transpose_b)
                _AddTest(MatMulTest, "MatMulTest", name,
                         _GetMatMulTest(
                             a_np,
                             b_np,
                             use_static_shape,
                             adjoint_a=adjoint_a,
                             transpose_a=transpose_a,
                             adjoint_b=adjoint_b,
                             transpose_b=transpose_b))
                _AddTest(MatMulGradientTest, "MatMulGradientTest", name,
                         _GetMatMulGradientTest(
                             a_np,
                             b_np,
                             use_static_shape,
                             adjoint_a=adjoint_a,
                             transpose_a=transpose_a,
                             adjoint_b=adjoint_b,
                             transpose_b=transpose_b))

  test_lib.main()
=======
"""Tests for tensorflow.ops.math_ops.matmul."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker as gc


class MatMulTest(tf.test.TestCase):

  def _testCpuMatmul(self, x, y, transpose_x=False, transpose_y=False):
    x_mat = np.matrix(x).T if transpose_x else np.matrix(x)
    y_mat = np.matrix(y).T if transpose_y else np.matrix(y)
    np_ans = x_mat * y_mat
    with self.test_session(use_gpu=False):
      tf_ans = tf.matmul(x, y, transpose_x, transpose_y).eval()
    self.assertAllClose(np_ans, tf_ans)
    self.assertAllEqual(np_ans.shape, tf_ans.shape)

  def _testGpuMatmul(self, x, y, transpose_x=False, transpose_y=False):
    x_mat = np.matrix(x).T if transpose_x else np.matrix(x)
    y_mat = np.matrix(y).T if transpose_y else np.matrix(y)
    np_ans = x_mat * y_mat
    with self.test_session(use_gpu=True):
      tf_ans = tf.matmul(x, y, transpose_x, transpose_y).eval()
    self.assertAllClose(np_ans, tf_ans)
    self.assertAllEqual(np_ans.shape, tf_ans.shape)

  def _randMatrix(self, rows, cols, dtype):
    if dtype is np.complex64:
      real = self._randMatrix(rows, cols, np.float32)
      imag = self._randMatrix(rows, cols, np.float32)
      return real + np.complex(0, 1) * imag
    else:
      return np.random.uniform(low=1.0, high=100.0, size=rows * cols).reshape(
          [rows, cols]).astype(dtype)

  # Basic test:
  #   [ [1],
  #     [2],
  #     [3],   *  [1, 2]
  #     [4] ]
  def testFloatBasic(self):
    x = np.arange(1., 5.).reshape([4, 1]).astype(np.float32)
    y = np.arange(1., 3.).reshape([1, 2]).astype(np.float32)
    self._testCpuMatmul(x, y)
    self._testGpuMatmul(x, y)

  def testDoubleBasic(self):
    x = np.arange(1., 5.).reshape([4, 1]).astype(np.float64)
    y = np.arange(1., 3.).reshape([1, 2]).astype(np.float64)
    self._testCpuMatmul(x, y)

  def testInt32Basic(self):
    x = np.arange(1., 5.).reshape([4, 1]).astype(np.int32)
    y = np.arange(1., 3.).reshape([1, 2]).astype(np.int32)
    self._testCpuMatmul(x, y)

  def testSComplexBasic(self):
    x = np.arange(1., 5.).reshape([4, 1]).astype(np.complex64)
    y = np.arange(1., 3.).reshape([1, 2]).astype(np.complex64)
    self._testCpuMatmul(x, y)

  # Tests testing random sized matrices.
  def testFloatRandom(self):
    for _ in range(10):
      n, k, m = np.random.randint(1, 100, size=3)
      x = self._randMatrix(n, k, np.float32)
      y = self._randMatrix(k, m, np.float32)
      self._testCpuMatmul(x, y)
      self._testGpuMatmul(x, y)

  def testDoubleRandom(self):
    for _ in range(10):
      n, k, m = np.random.randint(1, 100, size=3)
      x = self._randMatrix(n, k, np.float64)
      y = self._randMatrix(k, m, np.float64)
      self._testCpuMatmul(x, y)

  def testInt32Random(self):
    for _ in range(10):
      n, k, m = np.random.randint(1, 100, size=3)
      x = self._randMatrix(n, k, np.int32)
      y = self._randMatrix(k, m, np.int32)
      self._testCpuMatmul(x, y)

  def testSComplexRandom(self):
    for _ in range(10):
      n, k, m = np.random.randint(1, 100, size=3)
      x = self._randMatrix(n, k, np.complex64)
      y = self._randMatrix(k, m, np.complex64)
      self._testCpuMatmul(x, y)

  # Test the cases that transpose the matrices before multiplying.
  # NOTE(keveman): The cases where only one of the inputs is
  # transposed are covered by tf.matmul's gradient function.
  def testFloatRandomTransposeBoth(self):
    for _ in range(10):
      n, k, m = np.random.randint(1, 100, size=3)
      x = self._randMatrix(k, n, np.float32)
      y = self._randMatrix(m, k, np.float32)
      self._testCpuMatmul(x, y, True, True)
      self._testGpuMatmul(x, y, True, True)

  def testDoubleRandomTranposeBoth(self):
    for _ in range(10):
      n, k, m = np.random.randint(1, 100, size=3)
      x = self._randMatrix(k, n, np.float64)
      y = self._randMatrix(m, k, np.float64)
      self._testCpuMatmul(x, y, True, True)

  def testMatMul_OutEmpty_A(self):
      n, k, m = 0, 8, 3
      x = self._randMatrix(n, k, np.float32)
      y = self._randMatrix(k, m, np.float32)
      self._testCpuMatmul(x, y)
      self._testGpuMatmul(x, y)

  def testMatMul_OutEmpty_B(self):
      n, k, m = 3, 8, 0
      x = self._randMatrix(n, k, np.float32)
      y = self._randMatrix(k, m, np.float32)
      self._testCpuMatmul(x, y)
      self._testGpuMatmul(x, y)

  def testMatMul_Inputs_Empty(self):
      n, k, m = 3, 0, 4
      x = self._randMatrix(n, k, np.float32)
      y = self._randMatrix(k, m, np.float32)
      self._testCpuMatmul(x, y)
      self._testGpuMatmul(x, y)


# TODO(zhifengc): Figures out how to test matmul gradients on GPU.
class MatMulGradientTest(tf.test.TestCase):

  def testGradientInput0(self):
    with self.test_session(use_gpu=False):
      x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2],
                   dtype=tf.float64, name="x")
      y = tf.constant([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                   shape=[2, 4], dtype=tf.float64, name="y")
      m = tf.matmul(x, y, name="matmul")
      err = gc.ComputeGradientError(x, [3, 2], m, [3, 4])
    print "matmul input0 gradient err = ", err
    self.assertLess(err, 1e-10)

  def testGradientInput1(self):
    with self.test_session(use_gpu=False):
      x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2],
                   dtype=tf.float64, name="x")
      y = tf.constant([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                   shape=[2, 4], dtype=tf.float64, name="y")
      m = tf.matmul(x, y, name="matmul")
      err = gc.ComputeGradientError(y, [2, 4], m, [3, 4])
    print "matmul input1 gradient err = ", err
    self.assertLess(err, 1e-10)

  def _VerifyInput0(self, transpose_a, transpose_b):
    shape_x = [3, 2]
    shape_y = [2, 4]
    if transpose_a:
      shape_x = list(reversed(shape_x))
    if transpose_b:
      shape_y = list(reversed(shape_y))
    with self.test_session(use_gpu=False):
      x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=shape_x,
                   dtype=tf.float64, name="x")
      y = tf.constant([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                   shape=shape_y, dtype=tf.float64, name="y")
      m = tf.matmul(x, y, transpose_a, transpose_b, name="matmul")
      err = gc.ComputeGradientError(x, shape_x, m, [3, 4])
    print "matmul input0 gradient err = ", err
    self.assertLess(err, 1e-10)

  def testGradientInput0WithTranspose(self):
    self._VerifyInput0(transpose_a=True, transpose_b=False)
    self._VerifyInput0(transpose_a=False, transpose_b=True)
    self._VerifyInput0(transpose_a=True, transpose_b=True)

  def _VerifyInput1(self, transpose_a, transpose_b):
    shape_x = [3, 2]
    shape_y = [2, 4]
    if transpose_a:
      shape_x = list(reversed(shape_x))
    if transpose_b:
      shape_y = list(reversed(shape_y))
    with self.test_session(use_gpu=False):
      x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=shape_x,
                   dtype=tf.float64, name="x")
      y = tf.constant([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                   shape=shape_y, dtype=tf.float64, name="y")
      m = tf.matmul(x, y, transpose_a, transpose_b, name="matmul")
      err = gc.ComputeGradientError(y, shape_y, m, [3, 4])
    print "matmul input1 gradient err = ", err
    self.assertLess(err, 1e-10)

  def testGradientInput1WithTranspose(self):
    self._VerifyInput1(transpose_a=True, transpose_b=False)
    self._VerifyInput1(transpose_a=False, transpose_b=True)
    self._VerifyInput1(transpose_a=True, transpose_b=True)


if __name__ == "__main__":
  tf.test.main()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
