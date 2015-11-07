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
"""Functional tests for coefficient-wise operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.compat import compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
=======
"""Functional tests for coefficient-wise operations.
"""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker as gc
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

_ADD = lambda x, y: x + y
_SUB = lambda x, y: x - y
_MUL = lambda x, y: x * y
<<<<<<< HEAD
_POW = lambda x, y: x**y
_TRUEDIV = lambda x, y: x / y
_FLOORDIV = lambda x, y: x // y
_MOD = lambda x, y: x % y
=======
_DIV = lambda x, y: x / y
_MOD = lambda x, y: x % y
_NEG = lambda x: -x
_ABS = abs
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

_LT = lambda x, y: x < y
_LE = lambda x, y: x <= y
_GT = lambda x, y: x > y
_GE = lambda x, y: x >= y

_AND = lambda x, y: x & y
_OR = lambda x, y: x | y
_XOR = lambda x, y: x ^ y
_INV = lambda x: ~x


<<<<<<< HEAD
# TODO(zongheng): it'd be great to factor out this function and various random
# SparseTensor gen funcs.
def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), x_values


def _default_tolerance(dtype):
  """Returns a sensible default tolerance for comparing results of a given type.

  Args:
    dtype: A datatype.
  """
  if dtype == np.float16:
    return 5e-3
  elif dtype in (np.float32, np.complex64):
    return 1e-3
  elif dtype in (np.float64, np.complex128):
    return 1e-5
  else:
    return None  # Fail fast for unexpected types


class ComparisonOpTest(test.TestCase):

  def _compareScalar(self, func, x, y, dtype):
    with test_util.use_gpu():
      out = func(
          ops.convert_to_tensor(np.array([x]).astype(dtype)),
          ops.convert_to_tensor(np.array([y]).astype(dtype)))
      ret = self.evaluate(out)
    return ret[0]

  def testScalarCompareScalar(self):
    dtypes = [np.float16, np.float32, np.float64, np.int32, np.int64]
=======
class UnaryOpTest(tf.test.TestCase):

  def _compareCpu(self, x, np_func, tf_func):
    np_ans = np_func(x)
    with self.test_session(use_gpu=False):
      inx = tf.convert_to_tensor(x)
      y = tf_func(inx)
      tf_cpu = y.eval()
      self.assertShapeEqual(np_ans, y)
      self.assertAllClose(np_ans, tf_cpu)
      if x.dtype == np.float32:
        s = list(np.shape(x))
        jacob_t, jacob_n = gc.ComputeGradient(inx, s, y, s, x_init_value=x)
        self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
      elif x.dtype == np.float64:
        s = list(np.shape(x))
        jacob_t, jacob_n = gc.ComputeGradient(inx, s, y, s, x_init_value=x)
        self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGpu(self, x, np_func, tf_func):
    np_ans = np_func(x)
    with self.test_session(use_gpu=True):
      result = tf_func(tf.convert_to_tensor(x))
      tf_gpu = result.eval()
    self.assertShapeEqual(np_ans, result)
    self.assertAllClose(np_ans, tf_gpu)
    # TODO(zhifengc/ke): make gradient checker work on GPU.

  def _compareBoth(self, x, np_func, tf_func):
    self._compareCpu(x, np_func, tf_func)
    self._compareGpu(x, np_func, tf_func)

  def _inv(self, x):
    return 1.0 / x

  def _rsqrt(self, x):
    return self._inv(np.sqrt(x))

  def _sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  def testFloatBasic(self):
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float32)
    y = (x + .5).astype(np.float32)     # no zero
    z = (x + 15.5).astype(np.float32)   # all positive
    self._compareBoth(x, np.abs, tf.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, tf.neg)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(y, self._inv, tf.inv)
    self._compareBoth(x, np.square, tf.square)
    self._compareBoth(z, np.sqrt, tf.sqrt)
    self._compareBoth(z, self._rsqrt, tf.rsqrt)
    self._compareBoth(x, np.exp, tf.exp)
    self._compareBoth(z, np.log, tf.log)
    self._compareBoth(x, np.tanh, tf.tanh)
    self._compareBoth(x, self._sigmoid, tf.sigmoid)
    self._compareBoth(y, np.sign, tf.sign)
    self._compareBoth(x, np.sin, tf.sin)
    self._compareBoth(x, np.cos, tf.cos)

  def testFloatTanhEdge(self):
    x = np.arange(40, 40 + 6).reshape(6).astype(np.float32)
    self._compareBoth(x, np.tanh, tf.tanh)
    x = np.arange(-40, -40 + 6).reshape(6).astype(np.float32)
    self._compareBoth(x, np.tanh, tf.tanh)

  def testFloatEmpty(self):
    x = np.empty((2, 0, 5), dtype=np.float32)
    self._compareBoth(x, np.abs, tf.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, tf.neg)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(x, self._inv, tf.inv)
    self._compareBoth(x, np.square, tf.square)
    self._compareBoth(x, np.sqrt, tf.sqrt)
    self._compareBoth(x, self._rsqrt, tf.rsqrt)
    self._compareBoth(x, np.exp, tf.exp)
    self._compareBoth(x, np.log, tf.log)
    self._compareBoth(x, np.tanh, tf.tanh)
    self._compareBoth(x, self._sigmoid, tf.sigmoid)
    self._compareBoth(x, np.sign, tf.sign)
    self._compareBoth(x, np.sin, tf.sin)
    self._compareBoth(x, np.cos, tf.cos)

  def testDoubleBasic(self):
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float64)
    y = (x + .5).astype(np.float64)    # no zero
    z = (x + 15.5).astype(np.float64)  # all positive
    self._compareBoth(x, np.abs, tf.abs)
    self._compareBoth(x, np.abs, _ABS)
    self._compareBoth(x, np.negative, tf.neg)
    self._compareBoth(x, np.negative, _NEG)
    self._compareBoth(y, self._inv, tf.inv)
    self._compareBoth(x, np.square, tf.square)
    self._compareBoth(z, np.sqrt, tf.sqrt)
    self._compareBoth(z, self._rsqrt, tf.rsqrt)
    self._compareBoth(x, np.exp, tf.exp)
    self._compareBoth(z, np.log, tf.log)
    self._compareBoth(x, np.tanh, tf.tanh)
    self._compareBoth(x, self._sigmoid, tf.sigmoid)
    self._compareBoth(y, np.sign, tf.sign)
    self._compareBoth(x, np.sin, tf.sin)
    self._compareBoth(x, np.cos, tf.cos)

  def testInt32Basic(self):
    x = np.arange(-6, 6, 2).reshape(1, 3, 2).astype(np.int32)
    self._compareCpu(x, np.abs, tf.abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareCpu(x, np.negative, tf.neg)
    self._compareCpu(x, np.negative, _NEG)
    self._compareCpu(x, np.square, tf.square)
    self._compareCpu(x, np.sign, tf.sign)

  def testInt64Basic(self):
    x = np.arange(
        -6 << 40, 6 << 40, 2 << 40).reshape(1, 3, 2).astype(np.int64)
    self._compareCpu(x, np.abs, tf.abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareCpu(x, np.negative, tf.neg)
    self._compareCpu(x, np.negative, _NEG)
    self._compareCpu(x, np.square, tf.square)
    self._compareCpu(x, np.sign, tf.sign)

  def testComplex64Basic(self):
    x = np.complex(1, 1) * np.arange(-3, 3).reshape(1, 3, 2).astype(
        np.complex64)
    y = x + 0.5  # no zeros
    self._compareCpu(x, np.abs, tf.abs)
    self._compareCpu(x, np.abs, _ABS)
    self._compareCpu(x, np.negative, tf.neg)
    self._compareCpu(x, np.negative, _NEG)
    self._compareCpu(y, self._inv, tf.inv)
    self._compareCpu(x, np.square, tf.square)
    self._compareCpu(x, np.sqrt, tf.sqrt)
    self._compareCpu(y, self._rsqrt, tf.rsqrt)
    self._compareCpu(x, np.exp, tf.exp)
    self._compareCpu(y, np.log, tf.log)
    self._compareCpu(x, np.tanh, tf.tanh)
    self._compareCpu(x, self._sigmoid, tf.sigmoid)
    self._compareCpu(x, np.sin, tf.sin)
    self._compareCpu(x, np.cos, tf.cos)


class BinaryOpTest(tf.test.TestCase):

  def _compareCpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=False):
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_cpu = out.eval()
      # Test that the op takes precedence over numpy operators.
      np_left = tf_func(x, iny).eval()
      np_right = tf_func(inx, y).eval()

    self.assertAllClose(np_ans, tf_cpu)
    self.assertAllClose(np_ans, np_left)
    self.assertAllClose(np_ans, np_right)
    self.assertShapeEqual(np_ans, out)

  def _compareGradientX(self, x, y, np_func, tf_func):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      xs = list(x.shape)
      jacob_t, jacob_n = gc.ComputeGradient(inx, xs, out, zs, x_init_value=x)
      if x.dtype == np.float32:
        self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
      elif x.dtype == np.float64:
        self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, x, y, np_func, tf_func):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      ys = list(np.shape(y))
      jacob_t, jacob_n = gc.ComputeGradient(iny, ys, out, zs, x_init_value=y)
    if x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=True):
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = out.eval()
    self.assertAllClose(np_ans, tf_gpu)
    self.assertShapeEqual(np_ans, out)
    # TODO(zhifengc/ke): make gradient checker work on GPU.

  def _compareBoth(self, x, y, np_func, tf_func):
    self._compareCpu(x, y, np_func, tf_func)
    if x.dtype == np.float32 or x.dtype == np.float64:
      self._compareGradientX(x, y, np_func, tf_func)
      self._compareGradientY(x, y, np_func, tf_func)
      self._compareGpu(x, y, np_func, tf_func)

  def testFloatBasic(self):
    x = np.linspace(-10, 10, 6).reshape(1, 3, 2).astype(np.float32)
    y = np.linspace(20, -20, 6).reshape(1, 3, 2).astype(np.float32)
    self._compareBoth(x, y, np.add, tf.add)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y + 0.1, np.divide, tf.div)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y + 0.1, np.divide, _DIV)

  def testFloatDifferentShapes(self):
    x = np.array([1, 2, 3, 4]).reshape(2, 2).astype(np.float32)
    y = np.array([1, 2]).reshape(2, 1).astype(np.float32)
    with self.test_session() as sess:
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      s = tf.reduce_sum(inx * iny)
      gx, gy = sess.run(tf.gradients(s, [inx, iny]))
    # gx is simply the broadcasted y
    self.assertAllEqual(gx, np.array([1, 1, 2, 2])
                        .reshape(2, 2).astype(np.float32))
    # gy is x's column summed up
    self.assertAllEqual(gy, np.array([3, 7]).
                        reshape(2, 1).astype(np.float32))

  def testDoubleBasic(self):
    x = np.linspace(-10, 10, 6).reshape(1, 3, 2).astype(np.float64)
    y = np.linspace(20, -20, 6).reshape(1, 3, 2).astype(np.float64)
    self._compareBoth(x, y, np.add, tf.add)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y + 0.1, np.divide, tf.div)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y + 0.1, np.divide, _DIV)

  def testInt8Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int8)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int8)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y, np.multiply, _MUL)

  def testInt16Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int16)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int16)
    self._compareBoth(x, y, np.multiply, tf.mul)
    self._compareBoth(x, y, np.multiply, _MUL)

  def testInt32Basic(self):
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int32)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int32)
    self._compareBoth(x, y, np.add, tf.add)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    # NOTE: int32 division is ill-defined.
    self._compareBoth(x, y, np.divide, tf.div)
    self._compareBoth(x, y, np.mod, tf.mod)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    # NOTE: int32 division is ill-defined.
    self._compareBoth(x, y, np.divide, _DIV)
    self._compareBoth(x, y, np.mod, _MOD)

  def testInt64Basic(self):
    x = np.arange(1 << 40, 13 << 40, 2 << 40).reshape(1, 3, 2).astype(np.int64)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int64)
    self._compareBoth(x, y, np.subtract, tf.sub)
    self._compareBoth(x, y, np.multiply, tf.mul)
    # NOTE: int64 division is ill-defined.
    self._compareBoth(x, y, np.divide, tf.div)
    self._compareBoth(x, y, np.mod, tf.mod)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    # NOTE: int64 division is ill-defined.
    self._compareBoth(x, y, np.divide, _DIV)
    self._compareBoth(x, y, np.mod, _MOD)

  def testComplex64Basic(self):
    x = np.complex(1, 1) * np.linspace(-10, 10, 6).reshape(1, 3, 2).astype(
        np.complex64)
    y = np.complex(1, 1) * np.linspace(20, -20, 6).reshape(1, 3, 2).astype(
        np.complex64)
    self._compareCpu(x, y, np.add, tf.add)
    self._compareCpu(x, y, np.subtract, tf.sub)
    self._compareCpu(x, y, np.multiply, tf.mul)
    self._compareCpu(x, y + 0.1, np.divide, tf.div)
    self._compareCpu(x, y, np.add, _ADD)
    self._compareCpu(x, y, np.subtract, _SUB)
    self._compareCpu(x, y, np.multiply, _MUL)
    self._compareCpu(x, y + 0.1, np.divide, _DIV)

  def _compareBCast(self, xs, ys, dtype, np_func, tf_func):
    x = (1 + np.linspace(0, 5, np.prod(xs))).astype(dtype).reshape(xs)
    y = (1 + np.linspace(0, 5, np.prod(ys))).astype(dtype).reshape(ys)
    self._compareCpu(x, y, np_func, tf_func)
    if x.dtype == np.float32 or x.dtype == np.float64:
      self._compareGradientX(x, y, np_func, tf_func)
      self._compareGradientY(x, y, np_func, tf_func)
      self._compareGpu(x, y, np_func, tf_func)

  # TODO(josh11b,vrv): Refactor this to use parameterized tests.
  def _testBCastByFunc(self, funcs, xs, ys):
    dtypes = [
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.complex64
    ]
    for dtype in dtypes:
      for (np_func, tf_func) in funcs:
        self._compareBCast(xs, ys, dtype, np_func, tf_func)
        self._compareBCast(ys, xs, dtype, np_func, tf_func)

  def _testBCastA(self, xs, ys):
    funcs = [
        (np.add, tf.add),
        (np.add, _ADD),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastB(self, xs, ys):
    funcs = [
        (np.subtract, tf.sub),
        (np.subtract, _SUB),
        (np.power, tf.pow),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastC(self, xs, ys):
    funcs = [
        (np.multiply, tf.mul),
        (np.multiply, _MUL),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastD(self, xs, ys):
    funcs = [
        (np.divide, tf.div),
        (np.divide, _DIV)
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def testBCast_0A(self):
    self._testBCastA([1, 3, 2], [1])

  def testBCast_0B(self):
    self._testBCastB([1, 3, 2], [1])

  def testBCast_0C(self):
    self._testBCastC([1, 3, 2], [1])

  def testBCast_0D(self):
    self._testBCastD([1, 3, 2], [1])

  def testBCast_1A(self):
    self._testBCastA([1, 3, 2], [2])

  def testBCast_1B(self):
    self._testBCastB([1, 3, 2], [2])

  def testBCast_1C(self):
    self._testBCastC([1, 3, 2], [2])

  def testBCast_1D(self):
    self._testBCastD([1, 3, 2], [2])

  def testBCast_2A(self):
    self._testBCastA([1, 3, 2], [3, 2])

  def testBCast_2B(self):
    self._testBCastB([1, 3, 2], [3, 2])

  def testBCast_2C(self):
    self._testBCastC([1, 3, 2], [3, 2])

  def testBCast_2D(self):
    self._testBCastD([1, 3, 2], [3, 2])

  def testBCast_3A(self):
    self._testBCastA([1, 3, 2], [3, 1])

  def testBCast_3B(self):
    self._testBCastB([1, 3, 2], [3, 1])

  def testBCast_3C(self):
    self._testBCastC([1, 3, 2], [3, 1])

  def testBCast_3D(self):
    self._testBCastD([1, 3, 2], [3, 1])

  def testBCast_4A(self):
    self._testBCastA([1, 3, 2], [1, 3, 2])

  def testBCast_4B(self):
    self._testBCastB([1, 3, 2], [1, 3, 2])

  def testBCast_4C(self):
    self._testBCastC([1, 3, 2], [1, 3, 2])

  def testBCast_4D(self):
    self._testBCastD([1, 3, 2], [1, 3, 2])

  def testBCast_5A(self):
    self._testBCastA([1, 3, 2], [2, 3, 1])

  def testBCast_5B(self):
    self._testBCastB([1, 3, 2], [2, 3, 1])

  def testBCast_5C(self):
    self._testBCastC([1, 3, 2], [2, 3, 1])

  def testBCast_5D(self):
    self._testBCastD([1, 3, 2], [2, 3, 1])

  def testBCast_6A(self):
    self._testBCastA([1, 3, 2], [2, 1, 1])

  def testBCast_6B(self):
    self._testBCastB([1, 3, 2], [2, 1, 1])

  def testBCast_6C(self):
    self._testBCastC([1, 3, 2], [2, 1, 1])

  def testBCast_6D(self):
    self._testBCastD([1, 3, 2], [2, 1, 1])

  def testBCast_7A(self):
    self._testBCastA([1, 3, 2], [1, 3, 1])

  def testBCast_7B(self):
    self._testBCastB([1, 3, 2], [1, 3, 1])

  def testBCast_7C(self):
    self._testBCastC([1, 3, 2], [1, 3, 1])

  def testBCast_7D(self):
    self._testBCastD([1, 3, 2], [1, 3, 1])

  def testBCast_8A(self):
    self._testBCastA([2, 1, 5], [2, 3, 1])

  def testBCast_8B(self):
    self._testBCastB([2, 1, 5], [2, 3, 1])

  def testBCast_8C(self):
    self._testBCastC([2, 1, 5], [2, 3, 1])

  def testBCast_8D(self):
    self._testBCastD([2, 1, 5], [2, 3, 1])

  def testBCast_9A(self):
    self._testBCastA([2, 0, 5], [2, 0, 1])

  def testBCast_9B(self):
    self._testBCastB([2, 0, 5], [2, 0, 1])

  def testBCast_9C(self):
    self._testBCastC([2, 0, 5], [2, 0, 1])

  def testBCast_9D(self):
    self._testBCastD([2, 0, 5], [2, 0, 1])

  def testBCast_10A(self):
    self._testBCastA([2, 3, 0], [2, 3, 1])

  def testBCast_10B(self):
    self._testBCastB([2, 3, 0], [2, 3, 1])

  def testBCast_10C(self):
    self._testBCastC([2, 3, 0], [2, 3, 1])

  def testBCast_10D(self):
    self._testBCastD([2, 3, 0], [2, 3, 1])

  def testBCast_11A(self):
    self._testBCastA([1, 3, 2], [1, 3, 2])

  def testBCast_11B(self):
    self._testBCastB([1, 3, 2], [1, 3, 2])

  def testBCast_11C(self):
    self._testBCastC([1, 3, 2], [1, 3, 2])

  def testBCast_11D(self):
    self._testBCastD([1, 3, 2], [1, 3, 2])

  def testBCast_12A(self):
    self._testBCastA([1, 1, 1, 1, 3, 2], [1, 3, 2])

  def testBCast_12B(self):
    self._testBCastB([1, 1, 1, 1, 3, 2], [1, 3, 2])

  def testBCast_12C(self):
    self._testBCastC([1, 1, 1, 1, 3, 2], [1, 3, 2])

  def testBCast_12D(self):
    self._testBCastD([1, 1, 1, 1, 3, 2], [1, 3, 2])

  def testBCast_13A(self):
    self._testBCastA([1, 3, 2, 1, 1], [1])

  def testBCast_13B(self):
    self._testBCastB([1, 3, 2, 1, 1], [1])

  def testBCast_13C(self):
    self._testBCastC([1, 3, 2, 1, 1], [1])

  def testBCast_13D(self):
    self._testBCastD([1, 3, 2, 1, 1], [1])

  def testBCast_14A(self):
    self._testBCastA([2, 3, 1, 1, 5], [1])

  def testBCast_14B(self):
    self._testBCastB([2, 3, 1, 1, 5], [1])

  def testBCast_14C(self):
    self._testBCastC([2, 3, 1, 1, 5], [1])

  def testBCast_14D(self):
    self._testBCastD([2, 3, 1, 1, 5], [1])

  def testBCast_15A(self):
    self._testBCastA([10, 3, 1, 2], [3, 1, 2])

  def testBCast_15B(self):
    self._testBCastB([10, 3, 1, 2], [3, 1, 2])

  def testBCast_15C(self):
    self._testBCastC([10, 3, 1, 2], [3, 1, 2])

  def testBCast_15D(self):
    self._testBCastD([10, 3, 1, 2], [3, 1, 2])

  def testMismatchedDimensions(self):
    for func in [tf.add, tf.sub, tf.mul, tf.div,
                 _ADD, _SUB, _MUL, _DIV]:
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Incompatible shapes" in e.message):
        func(tf.convert_to_tensor([10.0, 20.0, 30.0]),
             tf.convert_to_tensor([[40.0, 50.0], [60.0, 70.0]]))


class ComparisonOpTest(tf.test.TestCase):

  def _compare(self, func, x, y, dtype):
    with self.test_session(use_gpu=False):
      out = func(tf.convert_to_tensor(np.array([x]).astype(dtype)),
                 tf.convert_to_tensor(np.array([y]).astype(dtype)))
      ret = out.eval()
    return ret[0]

  def testScalarCompareScalar(self):
    dtypes = [np.float32, np.float64, np.int32, np.int64]
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    data = [-1, 0, 1]
    for t in dtypes:
      for x in data:
        for y in data:
<<<<<<< HEAD
          self.assertEqual(self._compareScalar(math_ops.less, x, y, t), x < y)
          self.assertEqual(
              self._compareScalar(math_ops.less_equal, x, y, t), x <= y)
          self.assertEqual(
              self._compareScalar(math_ops.greater, x, y, t), x > y)
          self.assertEqual(
              self._compareScalar(math_ops.greater_equal, x, y, t), x >= y)
          self.assertEqual(self._compareScalar(math_ops.equal, x, y, t), x == y)
          self.assertEqual(
              self._compareScalar(math_ops.not_equal, x, y, t), x != y)
    data = [-1, 0, 1, -1j, 1j, 1 + 1j, 1 - 1j]
    for t in [np.complex64, np.complex128]:
      for x in data:
        for y in data:
          self.assertEqual(self._compareScalar(math_ops.equal, x, y, t), x == y)
          self.assertEqual(
              self._compareScalar(math_ops.not_equal, x, y, t), x != y)

  def _compare(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with test_util.use_gpu():
      out = tf_func(ops.convert_to_tensor(x), ops.convert_to_tensor(y))
      tf_ans = self.evaluate(out)
    self.assertAllEqual(np_ans, tf_ans)
=======
          self.assertEqual(self._compare(tf.less, x, y, t),
                           x < y)
          self.assertEqual(self._compare(tf.less_equal, x, y, t),
                           x <= y)
          self.assertEqual(self._compare(tf.greater, x, y, t),
                           x > y)
          self.assertEqual(self._compare(tf.greater_equal, x, y, t),
                           x >= y)
          self.assertEqual(self._compare(tf.equal, x, y, t),
                           x == y)
          self.assertEqual(self._compare(tf.not_equal, x, y, t),
                           x != y)

  def _compareCpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=False):
      out = tf_func(tf.convert_to_tensor(x), tf.convert_to_tensor(y))
      tf_cpu = out.eval()
    self.assertAllEqual(np_ans, tf_cpu)

  def _compareGpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=True):
      out = tf_func(tf.convert_to_tensor(x), tf.convert_to_tensor(y))
      tf_gpu = out.eval()
    self.assertAllEqual(np_ans, tf_gpu)

  def _compareBoth(self, x, y, np_func, tf_func):
    self._compareCpu(x, y, np_func, tf_func)
    if x.dtype == np.float32 or x.dtype == np.float64:
      self._compareGpu(x, y, np_func, tf_func)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def testTensorCompareTensor(self):
    x = np.linspace(-15, 15, 6).reshape(1, 3, 2)
    y = np.linspace(20, -10, 6).reshape(1, 3, 2)
<<<<<<< HEAD
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(xt, yt, np.less, math_ops.less)
      self._compare(xt, yt, np.less_equal, math_ops.less_equal)
      self._compare(xt, yt, np.greater, math_ops.greater)
      self._compare(xt, yt, np.greater_equal, math_ops.greater_equal)
      self._compare(xt, yt, np.equal, math_ops.equal)
      self._compare(xt, yt, np.not_equal, math_ops.not_equal)
    # Complex types do not support ordering but do support equality tests.
    for t in [np.complex64, np.complex128]:
      xt = x.astype(t)
      xt -= 1j * xt
      yt = y.astype(t)
      yt -= 1j * yt
      self._compare(xt, yt, np.equal, math_ops.equal)
      self._compare(xt, yt, np.not_equal, math_ops.not_equal)
=======
    for t in [np.float32, np.float64, np.int32, np.int64]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compareBoth(xt, yt, np.less, tf.less)
      self._compareBoth(xt, yt, np.less_equal, tf.less_equal)
      self._compareBoth(xt, yt, np.greater, tf.greater)
      self._compareBoth(xt, yt, np.greater_equal, tf.greater_equal)
      self._compareBoth(xt, yt, np.equal, tf.equal)
      self._compareBoth(xt, yt, np.not_equal, tf.not_equal)
    # TODO(zhifengc): complex64 doesn't work on GPU yet.
    self._compareCpu(x.astype(np.complex64), y.astype(np.complex64),
                     np.equal, tf.equal)
    self._compareCpu(x.astype(np.complex64), y.astype(np.complex64),
                     np.not_equal, tf.not_equal)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def _compareBCast(self, xs, ys, dtype, np_func, tf_func):
    x = np.linspace(-15, 15, np.prod(xs)).astype(dtype).reshape(xs)
    y = np.linspace(20, -10, np.prod(ys)).astype(dtype).reshape(ys)
<<<<<<< HEAD
    if dtype in (np.complex64, np.complex128):
      x -= 1j * x
      y -= 1j * y
    self._compare(x, y, np_func, tf_func)
    self._compare(y, x, np_func, tf_func)

  def _testBCastByFunc(self, np_func, tf_func, include_complex=False):
=======
    self._compareCpu(x, y, np_func, tf_func)
    self._compareCpu(y, x, np_func, tf_func)
    if x.dtype == np.float32 or x.dtype == np.float64:
      self._compareGpu(x, y, np_func, tf_func)
      self._compareGpu(y, x, np_func, tf_func)

  def _testBCastByFunc(self, np_func, tf_func):
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    shapes = [
        ([1, 3, 2], [1]),
        ([1, 3, 2], [2]),
        ([1, 3, 2], [3, 2]),
        ([1, 3, 2], [3, 1]),
        ([1, 3, 2], [1, 3, 2]),
        ([1, 3, 2], [2, 3, 1]),
        ([1, 3, 2], [2, 1, 1]),
        ([1, 3, 2], [1, 3, 1]),
        ([2, 1, 5], [2, 3, 1]),
        ([2, 0, 5], [2, 0, 1]),
        ([2, 3, 0], [2, 3, 1]),
    ]
    dtypes = [
<<<<<<< HEAD
        np.float16,
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
        np.float32,
        np.float64,
        np.int32,
        np.int64,
    ]
<<<<<<< HEAD
    if include_complex:
      dtypes.extend([np.complex64, np.complex128])

=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    for (xs, ys) in shapes:
      for dtype in dtypes:
        self._compareBCast(xs, ys, dtype, np_func, tf_func)

  def testBCastLess(self):
<<<<<<< HEAD
    self._testBCastByFunc(np.less, math_ops.less)

  def testBCastLessEqual(self):
    self._testBCastByFunc(np.less_equal, math_ops.less_equal)

  def testBCastGreater(self):
    self._testBCastByFunc(np.greater, math_ops.greater)

  def testBCastGreaterEqual(self):
    self._testBCastByFunc(np.greater_equal, math_ops.greater_equal)

  def testBCastEqual(self):
    self._testBCastByFunc(np.equal, math_ops.equal, include_complex=True)

  def testBCastNotEqual(self):
    self._testBCastByFunc(
        np.not_equal, math_ops.not_equal, include_complex=True)

  def testShapeMismatch(self):
    dtypes = [np.float16, np.float32, np.float64, np.int32, np.int64]
    funcs = [
        math_ops.less, math_ops.less_equal, math_ops.greater,
        math_ops.greater_equal, math_ops.equal, math_ops.not_equal
    ]
=======
    self._testBCastByFunc(np.less, tf.less)

  def testBCastLessEqual(self):
    self._testBCastByFunc(np.less_equal, tf.less_equal)

  def testBCastGreater(self):
    self._testBCastByFunc(np.greater, tf.greater)

  def testBCastGreaterEqual(self):
    self._testBCastByFunc(np.greater_equal, tf.greater_equal)

  def testBCastEqual(self):
    self._testBCastByFunc(np.equal, tf.equal)

  def testBCastNotEqual(self):
    self._testBCastByFunc(np.not_equal, tf.not_equal)

  def testShapeMismatch(self):
    dtypes = [np.float32, np.float64, np.int32, np.int64]
    funcs = [tf.less, tf.less_equal, tf.greater,
             tf.greater_equal, tf.equal, tf.not_equal]
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    x = np.arange(0, 10).reshape([2, 5])
    y = np.arange(0, 10).reshape([5, 2])
    for t in dtypes:
      for f in funcs:
<<<<<<< HEAD
        with self.assertRaisesRegexp(
            (ValueError, errors.InvalidArgumentError),
            "Incompatible shapes|Dimensions must be equal"):
          f(x.astype(t), y.astype(t))


class LogicalOpTest(test.TestCase):

  def _compareBinary(self, x, y, np_func, tf_func, use_gpu=False):
    np_ans = np_func(x, y)
    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_val = self.evaluate(out)
    self.assertEqual(out.dtype, dtypes_lib.bool)
=======
        with self.assertRaisesWithPredicateMatch(
            ValueError, lambda e: "Incompatible shapes" in e.message):
          f(x.astype(t), y.astype(t))


class LogicalOpTest(tf.test.TestCase):

  def _compareBinary(self, x, y, np_func, tf_func, use_gpu=False):
    np_ans = np_func(x, y)
    with self.test_session(use_gpu=use_gpu):
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_val = out.eval()
    self.assertEqual(out.dtype, tf.bool)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    self.assertAllEqual(np_ans, tf_val)
    self.assertShapeEqual(np_ans, out)

  def _not(self, x, use_gpu=False):
    np_ans = np.logical_not(x)
<<<<<<< HEAD
    with test_util.device(use_gpu=use_gpu):
      out = math_ops.logical_not(ops.convert_to_tensor(x))
      tf_val = self.evaluate(out)
    self.assertEqual(out.dtype, dtypes_lib.bool)
=======
    with self.test_session(use_gpu=use_gpu):
      out = tf.logical_not(tf.convert_to_tensor(x))
      tf_val = out.eval()
    self.assertEqual(out.dtype, tf.bool)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    self.assertAllEqual(np_ans, tf_val)
    self.assertShapeEqual(np_ans, out)

  def testScalar(self):
    data = [np.array([True]), np.array([False])]
    for use_gpu in [True, False]:
      for x in data:
        self._not(x, use_gpu)
      for x in data:
        for y in data:
<<<<<<< HEAD
          self._compareBinary(x, y, np.logical_and, math_ops.logical_and,
                              use_gpu)
          self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
          self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor,
                              use_gpu)
=======
          self._compareBinary(
              x, y, np.logical_and, tf.logical_and, use_gpu)
          self._compareBinary(
              x, y, np.logical_or, tf.logical_or, use_gpu)
          self._compareBinary(
              x, y, np.logical_xor, tf.logical_xor, use_gpu)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def testTensor(self):
    x = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    y = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    for use_gpu in [True, False]:
      self._not(x, use_gpu)
<<<<<<< HEAD
      self._compareBinary(x, y, np.logical_and, math_ops.logical_and, use_gpu)
      self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
      self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor, use_gpu)
=======
      self._compareBinary(x, y, np.logical_and, tf.logical_and, use_gpu)
      self._compareBinary(x, y, np.logical_or, tf.logical_or, use_gpu)
      self._compareBinary(x, y, np.logical_xor, tf.logical_xor, use_gpu)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def testBCast(self):
    shapes = [
        ([1, 3, 2], [1]),
        ([1, 3, 2], [2]),
        ([1, 3, 2], [3, 2]),
        ([1, 3, 2], [3, 1]),
        ([1, 3, 2], [1, 3, 2]),
        ([1, 3, 2], [2, 3, 1]),
        ([1, 3, 2], [2, 1, 1]),
        ([1, 3, 2], [1, 3, 1]),
        ([2, 1, 5], [2, 3, 1]),
        ([2, 0, 5], [2, 0, 1]),
        ([2, 3, 0], [2, 3, 1]),
    ]
    for (xs, ys) in shapes:
      x = np.random.randint(0, 2, np.prod(xs)).astype(np.bool).reshape(xs)
      y = np.random.randint(0, 2, np.prod(ys)).astype(np.bool).reshape(ys)
      for use_gpu in [True, False]:
<<<<<<< HEAD
        self._compareBinary(x, y, np.logical_and, math_ops.logical_and, use_gpu)
        self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
        self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor, use_gpu)

  @test_util.run_deprecated_v1
  def testShapeMismatch(self):
    x = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    y = np.random.randint(0, 2, 6).astype(np.bool).reshape(3, 2, 1)
    for f in [math_ops.logical_and, math_ops.logical_or, math_ops.logical_xor]:
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Dimensions must" in str(e)):
        f(x, y)

  @test_util.run_deprecated_v1
  def testUsingAsPythonValueFails(self):
    # Ensure that we raise an error when the user attempts to treat a
    # `Tensor` as a Python `bool`.
    b = constant_op.constant(False)
    with self.assertRaises(TypeError):
      if b:
        pass

    x = constant_op.constant(3)
    y = constant_op.constant(4)
    with self.assertRaises(TypeError):
      if x > y:
        pass

    z = constant_op.constant(7)

    # The chained comparison should fail because Python computes `x <
    # y` and short-circuits the comparison with `z` if it is `False`.
    with self.assertRaises(TypeError):
      _ = x < y < z


class SelectOpTest(test.TestCase):

  def _compare(self, fn, c, x, y, use_gpu):
    np_ans = np.where(c, x, y)
    with test_util.device(use_gpu=use_gpu):
      out = fn(c, x, y)
      tf_ans = self.evaluate(out)
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, out)

  def _compareGradientX(self,
                        fn,
                        c,
                        x,
                        y,
                        numeric_gradient_type=None,
                        x_init_value=None):
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = fn(c, inx, iny)
      s = list(np.shape(c))
      if x_init_value is None:
        x_init_value = x
      if x.shape != y.shape:
        x_init_value = np.broadcast_to(y, x.shape)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, s, out, s, x_init_value=x_init_value)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = fn(c, inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, s, outf, s, x_init_value=xf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
=======
        self._compareBinary(x, y, np.logical_and, tf.logical_and, use_gpu)
        self._compareBinary(x, y, np.logical_or, tf.logical_or, use_gpu)
        self._compareBinary(x, y, np.logical_xor, tf.logical_xor, use_gpu)

  def testShapeMismatch(self):
    x = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    y = np.random.randint(0, 2, 6).astype(np.bool).reshape(3, 2, 1)
    for f in [tf.logical_and, tf.logical_or, tf.logical_xor]:
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Incompatible shapes" in e.message):
        f(x, y)


class SelectOpTest(tf.test.TestCase):

  def _compare(self, c, x, y, use_gpu):
    np_ans = np.where(c, x, y)
    with self.test_session(use_gpu=use_gpu):
      out = tf.select(c, x, y)
      tf_ans = out.eval()
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, out)

  def _compareGradientX(self, c, x, y):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf.select(c, inx, iny)
      s = list(np.shape(c))
      jacob_t, jacob_n = gc.ComputeGradient(inx, s, out, s, x_init_value=x)
    if x.dtype == np.float32:
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

<<<<<<< HEAD
  def _compareGradientY(self, fn, c, x, y, numeric_gradient_type=None):
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = fn(c, inx, iny)
      s = list(np.shape(c))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, s, out, s, x_init_value=x, delta=1.0)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = fn(c, inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inyf, s, outf, s, x_init_value=yf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
=======
  def _compareGradientY(self, c, x, y):
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf.select(c, inx, iny)
      s = list(np.shape(c))
      jacob_t, jacob_n = gc.ComputeGradient(iny, s, out, s, x_init_value=y)
    if x.dtype == np.float32:
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

<<<<<<< HEAD
  def _testScalar(self, fn):
    c = True
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(fn, c, xt, yt, use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._compare(fn, c, xt, yt, use_gpu=True)

  def testScalar(self):
    self._testScalar(array_ops.where)
    self._testScalar(array_ops.where_v2)

  def _testScalarBroadcast(self, fn, c, x, y):
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(fn, c, xt, yt, use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._compare(fn, c, xt, yt, use_gpu=True)

  def testScalarBroadcast(self):
    c = True
    # where_v2 only
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 1, 1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 1, 2) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 2) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(3, 2) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)

  def _testBasic(self, fn):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(fn, c, xt, yt, use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._compare(fn, c, xt, yt, use_gpu=True)

  def testBasic(self):
    self._testBasic(array_ops.where)
    self._testBasic(array_ops.where_v2)

  def _testBasicBroadcast(self, fn, c, x, y):
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(fn, c, xt, yt, use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._compare(fn, c, xt, yt, use_gpu=True)

  def testBasicBroadcast(self):
    c0 = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    c1 = np.random.randint(0, 2, 2).astype(np.bool).reshape(1, 1, 2)
    c2 = np.random.randint(0, 2, 3).astype(np.bool).reshape(1, 3, 1)
    c3 = np.random.randint(0, 2, 1).astype(np.bool).reshape(1, 1, 1)
    for c in [c0, c1, c2, c3]:
      # where_v2 only
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1, 1) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 3, 1) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1, 2) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 2) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(3, 2) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)

  def _testGradients(self, fn):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [np.float16, np.float32, np.float64]:
      xt = x.astype(t)
      yt = y.astype(t)
      if t == np.float16:
        # Compare fp16 theoretical gradients to fp32 numerical gradients,
        # since fp16 numerical gradients are too imprecise unless great
        # care is taken with choosing the inputs and the delta. This is
        # a weaker check (in particular, it does not test the op itself,
        # only its gradient), but it's much better than nothing.
        self._compareGradientX(fn, c, xt, yt, np.float)
        self._compareGradientY(fn, c, xt, yt, np.float)
      else:
        self._compareGradientX(fn, c, xt, yt)
        self._compareGradientY(fn, c, xt, yt)

  @test_util.run_deprecated_v1
  def testGradients(self):
    self._testGradients(array_ops.where)
    self._testGradients(array_ops.where_v2)

  @test_util.run_deprecated_v1
  def testGradientsBroadcast(self):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    for t in [np.float32, np.float64]:
      # where_v2 only
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1, 1) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 3, 1) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1, 2) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 2) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(3, 2) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))

  def _testShapeMismatch(self, fn):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(2, 5, 3) * 100
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      with self.assertRaises(ValueError):
        fn(c, xt, yt)

  @test_util.run_deprecated_v1
  def testShapeMismatch(self):
    self._testShapeMismatch(array_ops.where)
    self._testShapeMismatch(array_ops.where_v2)

  def _testEmptyTensor(self, fn):
    c = np.random.randint(0, 3, 0).astype(np.bool).reshape(1, 3, 0)
    x = np.random.rand(1, 3, 0) * 100
    y = np.random.rand(1, 3, 0) * 100
    z_expected = np.zeros((1, 3, 0), dtype=np.float32)
    with self.cached_session():
      xt = x.astype(np.float32)
      yt = y.astype(np.float32)
      z = fn(c, xt, yt).eval()
      self.assertAllEqual(z_expected, z)

  @test_util.run_deprecated_v1
  def testEmptyTensor(self):
    self._testEmptyTensor(array_ops.where)
    self._testEmptyTensor(array_ops.where_v2)

  def _testNan(self, fn):
    with self.cached_session():
      for c in False, True:
        for a in 7.0, np.nan:
          for b in 5.0, np.nan:
            x = fn(c, a, b).eval()
            y = a if c else b
            self.assertEqual(np.isnan(x), np.isnan(y))

  @test_util.run_deprecated_v1
  def testNan(self):
    """Verify that nans don't propagate where they shouldn't."""
    self._testNan(array_ops.where)
    self._testNan(array_ops.where_v2)


class BatchSelectOpTest(test.TestCase):
  """Test broadcasting of Select when 'c' is a vec and 't' &'e' are rank2+."""

  def _compare(self, c, x, y, use_gpu):
    np_ans = np.dstack(
        [x_i if c_i else y_i for c_i, x_i, y_i in zip(c, x, y)]).transpose(
            [2, 0, 1])
    with test_util.device(use_gpu=use_gpu):
      out = array_ops.where(c, x, y)
      tf_ans = self.evaluate(out)
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, out)

  def _compareGradientX(self, c, x, y, numeric_gradient_type=None):
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = array_ops.where(c, inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, s, out, s, x_init_value=x)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = array_ops.where(c, inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, s, outf, s, x_init_value=xf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, c, x, y, numeric_gradient_type=None):
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = array_ops.where(c, inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, s, out, s, x_init_value=y)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = array_ops.where(c, inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inyf, s, outf, s, x_init_value=yf)
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def testBasic(self):
    c = np.random.randint(0, 2, 16).astype(np.bool)
    x = np.random.rand(16, 2, 8) * 100
    y = np.random.rand(16, 2, 8) * 100
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(c, xt, yt, use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._compare(c, xt, yt, use_gpu=True)

  @test_util.run_deprecated_v1
  def testGradients(self):
    c = np.random.randint(0, 2, 16).astype(np.bool)
    x = np.random.rand(16, 2, 8) * 100
    y = np.random.rand(16, 2, 8) * 100
    for t in [np.float16, np.float32, np.float64]:
      xt = x.astype(t)
      yt = y.astype(t)
      if t == np.float16:
        # Compare fp16 theoretical gradients to fp32 numerical gradients,
        # since fp16 numerical gradients are too imprecise unless great
        # care is taken with choosing the inputs and the delta. This is
        # a weaker check (in particular, it does not test the op itself,
        # only its gradient), but it's much better than nothing.
        self._compareGradientX(c, xt, yt, np.float)
        self._compareGradientY(c, xt, yt, np.float)
      else:
        self._compareGradientX(c, xt, yt)
        self._compareGradientY(c, xt, yt)

  @test_util.run_deprecated_v1
  def testShapeMismatch(self):
    c = np.random.randint(0, 2, 8).astype(np.bool)
    x = np.random.rand(16, 3, 2) * 100
    y = np.random.rand(16, 3, 2) * 100
    for t in [
        np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64,
        np.complex128
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      with self.assertRaises(ValueError):
        array_ops.where(c, xt, yt)


class MinMaxOpTest(test.TestCase):

  def _compare(self, x, y, use_gpu):
    np_min, np_max = np.minimum(x, y), np.maximum(x, y)
    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      omin, omax = math_ops.minimum(inx, iny), math_ops.maximum(inx, iny)
      tf_min, tf_max = self.evaluate([omin, omax])
=======
  def testBasic(self):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [np.float32, np.float64, np.int32, np.int64, np.complex64]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(c, xt, yt, use_gpu=False)
      if t in [np.float32, np.float64]:
        self._compare(c, xt, yt, use_gpu=True)

  def testGradients(self):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [np.float32, np.float64]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compareGradientX(c, xt, yt)
      self._compareGradientY(c, xt, yt)

  def testShapeMismatch(self):
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(2, 5, 3) * 100
    for t in [np.float32, np.float64, np.int32, np.int64, np.complex64]:
      xt = x.astype(t)
      yt = y.astype(t)
      with self.assertRaises(ValueError):
        tf.select(c, xt, yt)


class MinMaxOpTest(tf.test.TestCase):

  def _compare(self, x, y, use_gpu):
    np_min, np_max = np.minimum(x, y), np.maximum(x, y)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      omin, omax = tf.minimum(inx, iny), tf.maximum(inx, iny)
      tf_min, tf_max = sess.run([omin, omax])
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    self.assertAllEqual(np_min, tf_min)
    self.assertAllEqual(np_max, tf_max)

  def testBasic(self):
    x = np.random.rand(1, 3, 2) * 100.
    y = np.random.rand(1, 3, 2) * 100.
<<<<<<< HEAD
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
=======
    for t in [np.float32, np.float64, np.int32, np.int64]:
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      self._compare(x.astype(t), y.astype(t), use_gpu=False)
      self._compare(x.astype(t), y.astype(t), use_gpu=True)

  def testDifferentShapes(self):
    x = np.random.rand(1, 3, 2) * 100.
    y = np.random.rand(2) * 100.  # should broadcast
<<<<<<< HEAD
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
=======
    for t in [np.float32, np.float64, np.int32, np.int64]:
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      self._compare(x.astype(t), y.astype(t), use_gpu=False)
      self._compare(x.astype(t), y.astype(t), use_gpu=True)

  def testScalar(self):
    x = np.random.rand(1, 3, 2) * 100.
<<<<<<< HEAD
    y = np.random.rand(1).item() * 100.  # should broadcast
=======
    y = np.asscalar(np.random.rand(1) * 100.)  # should broadcast
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    # dropped np.float64, int64 because TF automatically converts to 32 bit
    for t in [np.float32, np.int32]:
      self._compare(x.astype(t), t(y), use_gpu=False)
      self._compare(x.astype(t), t(y), use_gpu=True)

  def _compareGradientX(self, func, x, y):
<<<<<<< HEAD
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, s, out, s, x_init_value=x)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
=======
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gc.ComputeGradient(inx, s, out, s, x_init_value=x)
    if x.dtype == np.float32:
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, func, x, y):
<<<<<<< HEAD
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, s, out, s, x_init_value=y)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
=======
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gc.ComputeGradient(iny, s, out, s, x_init_value=y)
    if x.dtype == np.float32:
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

<<<<<<< HEAD
  @test_util.run_deprecated_v1
=======
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  def testGradients(self):
    x = np.random.rand(1, 3, 2) * 100.
    # ensure x != y
    y = x + (np.random.randint(2, size=x.shape) - .5) * 2  # -1 or +1
<<<<<<< HEAD
    self._compareGradientX(math_ops.maximum, x, y)
    self._compareGradientY(math_ops.maximum, x, y)
    self._compareGradientX(math_ops.minimum, x, y)
    self._compareGradientY(math_ops.minimum, x, y)


class MathOpsOverloadTest(test.TestCase):

  def _computeTensorAndLiteral(self, x, y, dtype, func):
    with test_util.force_cpu():
      inx = ops.convert_to_tensor(x, dtype=dtype)
      z = func(inx, y)  # Should use __add__, __sub__, etc.
      return self.evaluate(z)

  def _computeLiteralAndTensor(self, x, y, dtype, func):
    with test_util.force_cpu():
      iny = ops.convert_to_tensor(y, dtype=dtype)
      z = func(x, iny)  # Should use __radd__, __rsub__, etc.
      return self.evaluate(z)

  def _compareBinary(self, x, y, dtype, np_func, tf_func):
    np_ans = np_func(x, y).astype(dtype.as_numpy_dtype)
    self.assertAllClose(np_ans,
                        self._computeTensorAndLiteral(x, y, dtype, tf_func))
    self.assertAllClose(np_ans,
                        self._computeLiteralAndTensor(x, y, dtype, tf_func))

  def _compareUnary(self, x, dtype, np_func, tf_func):
    np_ans = np_func(x).astype(dtype.as_numpy_dtype)
    with test_util.force_cpu():
      self.assertAllClose(
          np_ans, self.evaluate(tf_func(ops.convert_to_tensor(x, dtype=dtype))))

  def testOverload(self):
    dtypes = [
        dtypes_lib.float16,
        dtypes_lib.float32,
        dtypes_lib.float64,
        dtypes_lib.int32,
        dtypes_lib.int64,
        dtypes_lib.complex64,
        dtypes_lib.complex128,
=======
    self._compareGradientX(tf.maximum, x, y)
    self._compareGradientY(tf.maximum, x, y)
    self._compareGradientX(tf.minimum, x, y)
    self._compareGradientY(tf.minimum, x, y)


class MathOpsOverloadTest(tf.test.TestCase):

  def _computeTensorAndLiteral(self, x, y, dtype, func):
    with self.test_session(use_gpu=False):
      inx = tf.convert_to_tensor(x, dtype=dtype)
      z = func(inx, y)  # Should use __add__, __sub__, etc.
      return z.eval()

  def _computeLiteralAndTensor(self, x, y, dtype, func):
    with self.test_session(use_gpu=False):
      iny = tf.convert_to_tensor(y, dtype=dtype)
      z = func(x, iny)  # Should use __radd__, __rsub__, etc.
      return z.eval()

  def _compareBinary(self, x, y, dtype, np_func, tf_func):
    np_ans = np_func(x, y)
    self.assertAllClose(np_ans, self._computeTensorAndLiteral(
        x, y, dtype, tf_func))
    self.assertAllClose(np_ans, self._computeLiteralAndTensor(
        x, y, dtype, tf_func))

  def _compareUnary(self, x, dtype, np_func, tf_func):
    np_ans = np_func(x)
    with self.test_session(use_gpu=False):
      self.assertAllClose(np_ans, tf_func(tf.convert_to_tensor(x, dtype=dtype)).eval())

  def testOverload(self):
    dtypes = [
        tf.float32,
        tf.float64,
        tf.int32,
        tf.int64,
        tf.complex64,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    ]
    funcs = [
        (np.add, _ADD),
        (np.subtract, _SUB),
        (np.multiply, _MUL),
<<<<<<< HEAD
        (np.power, _POW),
        (np.true_divide, _TRUEDIV),
        (np.floor_divide, _FLOORDIV),
    ]
    for dtype in dtypes:
      for np_func, tf_func in funcs:
        if dtype in (dtypes_lib.complex64,
                     dtypes_lib.complex128) and tf_func == _FLOORDIV:
          continue  # floordiv makes no sense for complex
        self._compareBinary(10, 5, dtype, np_func, tf_func)
    # Mod only works for int32 and int64.
    for dtype in [dtypes_lib.int32, dtypes_lib.int64]:
=======
        (np.divide, _DIV)
    ]
    for dtype in dtypes:
      for np_func, tf_func in funcs:
        self._compareBinary(10, 5, dtype, np_func, tf_func)
    # Mod only works for int32 and int64.
    for dtype in [tf.int32, tf.int64]:
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      self._compareBinary(10, 3, dtype, np.mod, _MOD)

  def testOverloadComparisons(self):
    dtypes = [
<<<<<<< HEAD
        dtypes_lib.float16,
        dtypes_lib.float32,
        dtypes_lib.float64,
        dtypes_lib.int32,
        dtypes_lib.int64,
=======
        tf.float32,
        tf.float64,
        tf.int32,
        tf.int64,
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    ]
    funcs = [
        (np.less, _LT),
        (np.less_equal, _LE),
        (np.greater, _GT),
        (np.greater_equal, _GE),
    ]
    for dtype in dtypes:
      for np_func, tf_func in funcs:
        self._compareBinary(10, 5, dtype, np_func, tf_func)
<<<<<<< HEAD
    logical_funcs = [(np.logical_and, _AND), (np.logical_or, _OR),
                     (np.logical_xor, _XOR), (np.equal, math_ops.equal),
                     (np.not_equal, math_ops.not_equal)]
    for np_func, tf_func in logical_funcs:
      self._compareBinary(True, False, dtypes_lib.bool, np_func, tf_func)
      self._compareBinary(True, True, dtypes_lib.bool, np_func, tf_func)
      self._compareBinary(False, False, dtypes_lib.bool, np_func, tf_func)
      self._compareBinary(False, True, dtypes_lib.bool, np_func, tf_func)
      self._compareBinary([True, True, False, False],
                          [True, False, True, False], dtypes_lib.bool, np_func,
                          tf_func)
    self._compareUnary(True, dtypes_lib.bool, np.logical_not, _INV)
    self._compareUnary(False, dtypes_lib.bool, np.logical_not, _INV)
    self._compareUnary([True, False], dtypes_lib.bool, np.logical_not, _INV)


class IsFiniteInfNanTest(test.TestCase):

  def _compare(self, x, use_gpu):
    np_finite, np_inf, np_nan = np.isfinite(x), np.isinf(x), np.isnan(x)
    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      ofinite, oinf, onan = math_ops.is_finite(inx), math_ops.is_inf(
          inx), math_ops.is_nan(inx)
      tf_finite, tf_inf, tf_nan = self.evaluate([ofinite, oinf, onan])
=======
    logical_funcs = [
        (np.logical_and, _AND),
        (np.logical_or, _OR),
        (np.logical_xor, _XOR),
    ]
    for np_func, tf_func in logical_funcs:
      self._compareBinary(True, False, tf.bool, np_func, tf_func)
      self._compareBinary(True, True, tf.bool, np_func, tf_func)
      self._compareBinary(False, False, tf.bool, np_func, tf_func)
      self._compareBinary(False, True, tf.bool, np_func, tf_func)
      self._compareBinary([True, True, False, False],
                          [True, False, True, False],
                          tf.bool, np_func, tf_func)
    self._compareUnary(True, tf.bool, np.logical_not, _INV)
    self._compareUnary(False, tf.bool, np.logical_not, _INV)
    self._compareUnary([True, False], tf.bool, np.logical_not, _INV)


class IsFiniteInfNanTest(tf.test.TestCase):

  def _compare(self, x, use_gpu):
    np_finite, np_inf, np_nan = np.isfinite(x), np.isinf(x), np.isnan(x)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(x)
      ofinite, oinf, onan = tf.is_finite(inx), tf.is_inf(
          inx), tf.is_nan(inx)
      tf_finite, tf_inf, tf_nan = sess.run([ofinite, oinf, onan])
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    self.assertAllEqual(np_inf, tf_inf)
    self.assertAllEqual(np_nan, tf_nan)
    self.assertAllEqual(np_finite, tf_finite)
    self.assertShapeEqual(np_inf, oinf)
    self.assertShapeEqual(np_nan, onan)
    self.assertShapeEqual(np_finite, ofinite)

  def _testDtype(self, dtype):
    fi = np.finfo(dtype)
<<<<<<< HEAD
    data = np.array([
        0, -1, 1, fi.resolution, -fi.resolution, fi.min, fi.max, -np.inf,
        np.inf, np.nan
    ]).astype(dtype)
    self._compare(data, use_gpu=False)
    self._compare(data, use_gpu=True)

  def testHalf(self):
    self._testDtype(np.float16)

=======
    data = np.array([0, -1, 1, fi.resolution, -fi.resolution, fi.min, fi.max,
                     -np.inf, np.inf, np.nan]).astype(dtype)
    self._compare(data, use_gpu=False)
    self._compare(data, use_gpu=True)

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  def testFloat(self):
    self._testDtype(np.float32)

  def testDouble(self):
    self._testDtype(np.float64)

<<<<<<< HEAD
  def testSqrt(self):
    for dtype in [np.float16, np.float32, np.float64]:
      fi = np.finfo(dtype)
      for size in [1, 3, 4, 7, 8, 63, 64, 65]:
        # For float32 Eigen uses Carmack's fast vectorized sqrt algorithm.
        # It is not accurate for very large arguments, so we test for
        # fi.max/100 instead of fi.max here.
        for value in [fi.min, -2, -1, 0, fi.tiny, 1, 2, 1000, fi.max / 100]:
          x = np.full((size,), value, dtype=dtype)
          np_y = np.sqrt(x)
          np_nan = np.isnan(np_y)
          with test_util.use_gpu():
            tf_y = math_ops.sqrt(x)
            tf_nan = math_ops.is_nan(tf_y)
            if value < 0:
              self.assertAllEqual(np_nan, self.evaluate(tf_nan))
            else:
              self.assertAllCloseAccordingToType(np_y, self.evaluate(tf_y))


class RoundingTest(test.TestCase):

  def _compare_values(self, x, y=None):
    y = np.rint(x) if y is None else np.asarray(y)

    tf_rint = math_ops.rint(x)
    np_rint = self.evaluate(tf_rint)

    self.assertAllEqual(y, np_rint)
    self.assertShapeEqual(y, tf_rint)

  def _compare(self, x):
    np_floor, np_ceil = np.floor(x), np.ceil(x)

    inx = ops.convert_to_tensor(x)
    ofloor, oceil = math_ops.floor(inx), math_ops.ceil(inx)
    tf_floor, tf_ceil = self.evaluate([ofloor, oceil])

=======

class RoundingTest(tf.test.TestCase):

  def _compare(self, x, use_gpu):
    np_floor, np_ceil = np.floor(x), np.ceil(x)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(x)
      ofloor, oceil = tf.floor(inx), tf.ceil(inx)
      tf_floor, tf_ceil = sess.run([ofloor, oceil])
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    self.assertAllEqual(np_floor, tf_floor)
    self.assertAllEqual(np_ceil, tf_ceil)
    self.assertShapeEqual(np_floor, ofloor)
    self.assertShapeEqual(np_ceil, oceil)

  def _testDtype(self, dtype):
<<<<<<< HEAD
    data = (np.arange(-3, 3) / 4.).reshape(1, 3, 2).astype(dtype)
    self._compare(data)
    # TODO: rint op is not supported for float16
    if dtype is np.float16:
      return
    self._compare_values(data)
    x = [0.5, 0.5000001]
    y = [0.0, 1.0]
    self._compare_values(x, y=y)

    # numpy example
    x = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]
    y = [-2., -2., -0., 0., 2., 2., 2.]
    self._compare_values(x, y=y)

  def testTypes(self):
    self.skipTest("b/131162241")
    for dtype in [np.float16, np.float32, np.float64]:
      self._testDtype(dtype)


class ComplexMakeRealImagTest(test.TestCase):

  def _compareMake(self, real, imag, use_gpu):
    np_ans = real + (1j) * imag

    with test_util.device(use_gpu=use_gpu):
      real = ops.convert_to_tensor(real)
      imag = ops.convert_to_tensor(imag)
      tf_ans = math_ops.complex(real, imag)
      out = self.evaluate(tf_ans)

=======
    data = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(dtype)
    self._compare(data, use_gpu=True)
    self._compare(data, use_gpu=True)

  def testTypes(self):
    for dtype in [np.float32, np.float64]:
      self._testDtype(dtype)


class ComplexMakeRealImagTest(tf.test.TestCase):

  def _compareMake(self, real, imag, use_gpu):
    np_ans = real + (1j) * imag
    with self.test_session(use_gpu=use_gpu):
      real = tf.convert_to_tensor(real)
      imag = tf.convert_to_tensor(imag)
      tf_ans = tf.complex(real, imag)
      out = tf_ans.eval()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def testMake(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    for use_gpu in [False, True]:
      self._compareMake(real, imag, use_gpu)
      self._compareMake(real, 12.0, use_gpu)
      self._compareMake(23.0, imag, use_gpu)

<<<<<<< HEAD
  def testRealImagNumericType(self):
    for use_gpu in [True, False]:
      for value in [1., 1j, 1. + 1j]:
        np_real, np_imag = np.real(value), np.imag(value)
        with test_util.device(use_gpu=use_gpu):
          tf_real = math_ops.real(value)
          tf_imag = math_ops.imag(value)
          self.assertAllEqual(np_real, self.evaluate(tf_real))
          self.assertAllEqual(np_imag, self.evaluate(tf_imag))

  def _compareRealImag(self, cplx, use_gpu):
    np_real, np_imag = np.real(cplx), np.imag(cplx)
    np_zeros = np_real * 0

    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(cplx)
      tf_real = math_ops.real(inx)
      tf_imag = math_ops.imag(inx)
      tf_real_real = math_ops.real(tf_real)
      tf_imag_real = math_ops.imag(tf_real)
      self.assertAllEqual(np_real, self.evaluate(tf_real))
      self.assertAllEqual(np_imag, self.evaluate(tf_imag))
      self.assertAllEqual(np_real, self.evaluate(tf_real_real))
      self.assertAllEqual(np_zeros, self.evaluate(tf_imag_real))

  def testRealImag64(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    cplx = real + 1j * imag
    self._compareRealImag(cplx, use_gpu=False)
    self._compareRealImag(cplx, use_gpu=True)

  def testRealImag128(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float64)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float64)
    cplx = real + 1j * imag
    self._compareRealImag(cplx, use_gpu=False)
    self._compareRealImag(cplx, use_gpu=True)

  def _compareAngle(self, cplx, use_gpu):
    np_angle = np.angle(cplx)

    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(cplx)
      tf_angle = math_ops.angle(inx)
      tf_angle_val = self.evaluate(tf_angle)

    self.assertAllClose(np_angle, tf_angle_val)
    self.assertShapeEqual(np_angle, tf_angle)

  def testAngle64(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    cplx = real + 1j * imag
    self._compareAngle(cplx, use_gpu=False)
    self._compareAngle(cplx, use_gpu=True)

  def testAngle(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float64)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float64)
    cplx = real + 1j * imag
    self._compareAngle(cplx, use_gpu=False)
    self._compareAngle(cplx, use_gpu=True)

  @test_util.run_deprecated_v1
  def testRealReal(self):
    for dtype in (dtypes_lib.int32, dtypes_lib.int64, dtypes_lib.float32,
                  dtypes_lib.float64):
      x = array_ops.placeholder(dtype)
      y = math_ops.real(x)
      self.assertEqual(x, y)

  def _compareConj(self, cplx, use_gpu):
    np_ans = np.conj(cplx)
    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(cplx)
      tf_conj = math_ops.conj(inx)
      tf_ans = self.evaluate(tf_conj)
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, tf_conj)

  def testConj64(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    cplx = real + 1j * imag
    self._compareConj(cplx, use_gpu=False)
    self._compareConj(cplx, use_gpu=True)

  def testConj128(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float64)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float64)
    cplx = real + 1j * imag
    self._compareConj(cplx, use_gpu=False)
    self._compareConj(cplx, use_gpu=True)

  @test_util.run_deprecated_v1
  def testConjReal(self):
    for dtype in (dtypes_lib.int32, dtypes_lib.int64, dtypes_lib.float16,
                  dtypes_lib.float32, dtypes_lib.float64):
      x = array_ops.placeholder(dtype)
      y = math_ops.conj(x)
      self.assertEqual(x, y)

  @test_util.run_deprecated_v1
  def testConjString(self):
    x = array_ops.placeholder(dtypes_lib.string)
    with self.assertRaisesRegexp(TypeError,
                                 r"Expected numeric or variant tensor"):
      math_ops.conj(x)

=======
  def _compareRealImag(self, cplx, use_gpu):
    np_real, np_imag = np.real(cplx), np.imag(cplx)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(cplx)
      tf_real = tf.real(inx)
      tf_imag = tf.imag(inx)
      tf_real_val, tf_imag_val = sess.run([tf_real, tf_imag])
    self.assertAllEqual(np_real, tf_real_val)
    self.assertAllEqual(np_imag, tf_imag_val)
    self.assertShapeEqual(np_real, tf_real)
    self.assertShapeEqual(np_imag, tf_imag)

  def testRealImag(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    cplx = real + (1j) * imag
    self._compareRealImag(cplx, use_gpu=False)
    self._compareRealImag(cplx, use_gpu=True)

  def _compareConj(self, cplx, use_gpu):
    np_ans = np.conj(cplx)
    with self.test_session(use_gpu=use_gpu):
      inx = tf.convert_to_tensor(cplx)
      tf_conj = tf.conj(inx)
      tf_ans = tf_conj.eval()
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, tf_conj)

  def testConj(self):
    real = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(np.float32)
    imag = (np.arange(-3, 3) / 5.).reshape([1, 3, 2]).astype(np.float32)
    cplx = real + (1j) * imag
    self._compareConj(cplx, use_gpu=False)
    self._compareConj(cplx, use_gpu=True)

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  def _compareGradient(self, x):
    # x[:, 0] is real, x[:, 1] is imag.  We combine real and imag into
    # complex numbers. Then, we extract real and imag parts and
    # computes the squared sum. This is obviously the same as sum(real
    # * real) + sum(imag * imag). We just want to make sure the
    # gradient function is checked.
<<<<<<< HEAD
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      real, imag = array_ops.split(value=inx, num_or_size_splits=2, axis=1)
      real, imag = array_ops.reshape(real, [-1]), array_ops.reshape(imag, [-1])
      cplx = math_ops.complex(real, imag)
      cplx = math_ops.conj(cplx)
      loss = math_ops.reduce_sum(math_ops.square(
          math_ops.real(cplx))) + math_ops.reduce_sum(
              math_ops.square(math_ops.imag(cplx)))
      epsilon = 1e-3
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, list(x.shape), loss, [1], x_init_value=x, delta=epsilon)
    self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

  def _compareBroadcastGradient(self, x):
    x_ = ops.convert_to_tensor(x)
    epsilon = 1e-3
    with self.cached_session():
      for args in [(x_, 0.), (0., x_)]:
        z = math_ops.reduce_sum(math_ops.abs(math_ops.complex(*args)))
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            x_, list(x.shape), z, [1], x_init_value=x, delta=epsilon)
        self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # complex64
    data = np.arange(1, 2, 0.10).reshape([5, 2]).astype(np.float32)
    self._compareGradient(data)
    self._compareBroadcastGradient(data)
    # complex128
    data = np.arange(1, 2, 0.10).reshape([5, 2]).astype(np.float64)
    self._compareGradient(data)
=======
    with self.test_session():
      inx = tf.convert_to_tensor(x)
      real, imag = tf.split(1, 2, inx)
      real, imag = tf.reshape(real, [-1]), tf.reshape(imag, [-1])
      cplx = tf.complex(real, imag)
      cplx = tf.conj(cplx)
      loss = tf.reduce_sum(
          tf.square(tf.real(cplx))) + tf.reduce_sum(
              tf.square(tf.imag(cplx)))
      epsilon = 1e-3
      jacob_t, jacob_n = gc.ComputeGradient(inx, list(x.shape), loss, [1],
                                            x_init_value=x, delta=epsilon)
    self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

  def testGradient(self):
    data = np.arange(1, 2, 0.10).reshape([5, 2]).astype(np.float32)
    self._compareGradient(data)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.

  def _compareMulGradient(self, data):
    # data is a float matrix of shape [n, 4].  data[:, 0], data[:, 1],
    # data[:, 2], data[:, 3] are real parts of x, imaginary parts of
    # x, real parts of y and imaginary parts of y.
<<<<<<< HEAD
    with self.cached_session():
      inp = ops.convert_to_tensor(data)
      xr, xi, yr, yi = array_ops.split(value=inp, num_or_size_splits=4, axis=1)

      def vec(x):  # Reshape to a vector
        return array_ops.reshape(x, [-1])

      xr, xi, yr, yi = vec(xr), vec(xi), vec(yr), vec(yi)

      def cplx(r, i):  # Combine to a complex vector
        return math_ops.complex(r, i)

=======
    with self.test_session():
      inp = tf.convert_to_tensor(data)
      xr, xi, yr, yi = tf.split(1, 4, inp)

      def vec(x):  # Reshape to a vector
        return tf.reshape(x, [-1])
      xr, xi, yr, yi = vec(xr), vec(xi), vec(yr), vec(yi)

      def cplx(r, i):  # Combine to a complex vector
        return tf.complex(r, i)
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
      x, y = cplx(xr, xi), cplx(yr, yi)
      # z is x times y in complex plane.
      z = x * y
      # Defines the loss function as the sum of all coefficients of z.
<<<<<<< HEAD
      loss = math_ops.reduce_sum(math_ops.real(z) + math_ops.imag(z))
      epsilon = 0.005
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inp, list(data.shape), loss, [1], x_init_value=data, delta=epsilon)
    self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

  @test_util.run_deprecated_v1
=======
      loss = tf.reduce_sum(tf.real(z) + tf.imag(z))
      epsilon = 0.005
      jacob_t, jacob_n = gc.ComputeGradient(inp, list(data.shape), loss, [1],
                                            x_init_value=data, delta=epsilon)
    self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
  def testMulGradient(self):
    data = np.arange(1, 2, 0.125).reshape([2, 4]).astype(np.float32)
    self._compareMulGradient(data)


<<<<<<< HEAD
class PolyvalTest(test.TestCase):

  def _runtest(self, dtype, degree):
    x = np.random.rand(2, 2).astype(dtype)
    coeffs = [np.random.rand(2, 2).astype(dtype) for _ in range(degree + 1)]
    np_val = np.polyval(coeffs, x)
    with self.cached_session():
      tf_val = math_ops.polyval(coeffs, x)
      self.assertAllClose(np_val, self.evaluate(tf_val))

  def testSimple(self):
    for dtype in [
        np.int32, np.float32, np.float64, np.complex64, np.complex128
    ]:
      for degree in range(5):
        self._runtest(dtype, degree)

  def testBroadcast(self):
    dtype = np.float32
    degree = 3
    shapes = [(1,), (2, 1), (1, 2), (2, 2)]
    for x_shape in shapes:
      for coeff_shape in shapes:
        x = np.random.rand(*x_shape).astype(dtype)
        coeffs = [
            np.random.rand(*coeff_shape).astype(dtype)
            for _ in range(degree + 1)
        ]
        np_val = np.polyval(coeffs, x)
        with self.cached_session():
          tf_val = math_ops.polyval(coeffs, x)
          self.assertAllClose(np_val, self.evaluate(tf_val))

  def testEmpty(self):
    x = np.random.rand(2, 2).astype(np.float32)
    coeffs = []
    np_val = np.polyval(coeffs, x)
    with self.cached_session():
      tf_val = math_ops.polyval(coeffs, x)
      self.assertAllClose(np_val, self.evaluate(tf_val))


class SingularGradientOpTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testGradientAtSingularity(self):
    if not compat.forward_compatible(2020, 3, 14):
      self.skipTest("Skipping test for future functionality.")

    ops_and_singularity = [
        (gen_math_ops.reciprocal, (0.,)),
        (gen_math_ops.rsqrt, (0.,)),
        (gen_math_ops.sqrt, (0.,)),
        (gen_math_ops.sqrt_grad, (
            0.,
            0.,
        )),
        (gen_math_ops.reciprocal_grad, (
            1.,
            0.,
        )),
        (gen_math_ops.tan, (np.pi / 2,)),
        (gen_math_ops.log, (0.,)),
        (gen_math_ops.log1p, (-1.,)),
        (gen_math_ops.acosh, (0.,)),
        (gen_math_ops.asin, (1.,)),
        (gen_math_ops.acos, (1.,)),
        (gen_math_ops.atan2, (0., 0.)),
        (gen_math_ops.div, (1., 0.)),
        (gen_math_ops.div_no_nan, (1., 0.)),
        (gen_math_ops.real_div, (1., 0.)),
        (math_ops.pow, (0., -1.)),
    ]
    for op, singularity in ops_and_singularity:
      for dtype in (dtypes_lib.half, dtypes_lib.float32, dtypes_lib.float64,
                    dtypes_lib.complex64, dtypes_lib.complex128):
        if dtype.is_complex and op in [
            gen_math_ops.asin, gen_math_ops.acos, gen_math_ops.atan2
        ]:
          continue
        if dtype == dtypes_lib.half and op in [
            gen_math_ops.acosh, gen_math_ops.asin, gen_math_ops.acos,
            gen_math_ops.atan2
        ]:
          continue
        with self.cached_session():
          print("op = ", op, ", singularity = ", singularity, ", type = ",
                dtype)
          args = [constant_op.constant(s, dtype=dtype) for s in singularity]
          grad_y = constant_op.constant(0, dtype=dtype)
          y = op(*args)
          g = gradients_impl.gradients(y, args, grad_ys=grad_y)
          g_val = self.evaluate(g)
          self.assertAllEqual(g_val, np.zeros(len(singularity)))


if __name__ == "__main__":
  test.main()
=======
class AccumulateTest(tf.test.TestCase):

  def testSimple(self):
    with self.test_session():
      random_arrays = [np.random.rand(16, 16, 16, 16).astype(np.float32)
                       for _ in range(20)]
      random_tensors = [tf.convert_to_tensor(x, dtype=tf.float32)
                        for x in random_arrays]
      tf_val = tf.accumulate_n(random_tensors)
      np_val = random_arrays[0]
      for random_array in random_arrays[1:]:
        np_val += random_array
      self.assertAllClose(np_val, tf_val.eval())

  def testZeroArgs(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf_val = tf.accumulate_n([])
        tf_val.eval()

if __name__ == "__main__":
  tf.test.main()
>>>>>>> f41959ccb2... TensorFlow: Initial commit of TensorFlow library.
